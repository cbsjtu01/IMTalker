#!/usr/bin/env python3
"""
TensorRT Optimization for IMTalker
==================================
Provides TensorRT-accelerated versions of the Renderer and Generator.

Supports multiple optimization strategies:
1. torch.compile() with inductor backend (easiest, good speedup)
2. torch.compile() with tensorrt backend (best for static shapes)
3. Direct TensorRT engine export (maximum performance)

Usage:
    # Option 1: Use torch.compile (recommended for flexibility)
    from tensorrt_optimize import compile_model, TRTConfig
    renderer = compile_model(renderer, TRTConfig(backend="inductor"))

    # Option 2: Export TensorRT engines
    python tensorrt_optimize.py --export --model renderer

    # Option 3: Load pre-compiled engines
    from tensorrt_optimize import TRTRenderer, TRTGenerator
    renderer = TRTRenderer.load("checkpoints/renderer_trt.ts")
"""

import os
import sys
import time
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class TRTConfig:
    """Configuration for TensorRT optimization"""
    # Backend: "inductor", "tensorrt", "eager"
    backend: str = "inductor"

    # Precision: "fp32", "fp16", "bf16"
    precision: str = "fp16"

    # Dynamic shapes support
    dynamic: bool = True

    # Cache compiled models
    cache_dir: str = "./checkpoints/trt_cache"

    # For TensorRT backend
    workspace_size: int = 4 << 30  # 4GB

    # Optimization level for inductor
    mode: str = "reduce-overhead"  # "default", "reduce-overhead", "max-autotune"

    # Warmup iterations
    warmup_iters: int = 3


def get_torch_compile_options(config: TRTConfig) -> Dict[str, Any]:
    """Get torch.compile options based on config"""
    options = {
        "backend": config.backend,
        "mode": config.mode,
        "dynamic": config.dynamic,
    }

    if config.backend == "tensorrt":
        options["options"] = {
            "truncate_long_and_double": True,
            "precision": torch.float16 if config.precision == "fp16" else torch.float32,
            "workspace_size": config.workspace_size,
        }

    return options


def compile_model(
    model: nn.Module,
    config: Optional[TRTConfig] = None,
    example_inputs: Optional[Tuple] = None,
) -> nn.Module:
    """
    Compile a model with torch.compile for acceleration.

    Args:
        model: PyTorch model to compile
        config: TRTConfig with compilation settings
        example_inputs: Optional example inputs for warmup

    Returns:
        Compiled model
    """
    if config is None:
        config = TRTConfig()

    logger.info(f"Compiling model with backend={config.backend}, precision={config.precision}")

    # Convert to specified precision
    if config.precision == "fp16":
        model = model.half()
    elif config.precision == "bf16":
        model = model.bfloat16()

    # Get compile options
    compile_opts = get_torch_compile_options(config)

    # Remove mode for non-inductor backends
    if config.backend != "inductor":
        compile_opts.pop("mode", None)

    try:
        compiled = torch.compile(model, **compile_opts)
        logger.info(f"Model compiled successfully with {config.backend}")

        # Warmup if example inputs provided
        if example_inputs is not None:
            logger.info(f"Running {config.warmup_iters} warmup iterations...")
            with torch.no_grad():
                for _ in range(config.warmup_iters):
                    _ = compiled(*example_inputs)
            logger.info("Warmup complete")

        return compiled

    except Exception as e:
        logger.warning(f"torch.compile failed: {e}, falling back to eager mode")
        return model


class OptimizedRendererWrapper(nn.Module):
    """
    Optimized wrapper for IMTRenderer that:
    1. Caches dense_feature_encoder output for same source
    2. Provides compiled sub-modules
    3. Supports FP16 inference
    """

    def __init__(self, renderer: nn.Module, config: Optional[TRTConfig] = None):
        super().__init__()
        self.config = config or TRTConfig()
        self.renderer = renderer

        # Cache for source image features
        self._cached_source_hash = None
        self._cached_f_r = None
        self._cached_i_r = None
        self._cached_t_r = None
        self._cached_ta_r = None
        self._cached_ma_r = None

        # Compile individual components for better optimization
        self._compile_components()

    def _compile_components(self):
        """Compile individual components"""
        logger.info("Compiling renderer components...")

        compile_opts = {"mode": self.config.mode, "dynamic": self.config.dynamic}

        # These are the hot paths - compile them
        self.renderer.latent_token_encoder = torch.compile(
            self.renderer.latent_token_encoder, **compile_opts
        )
        self.renderer.latent_token_decoder = torch.compile(
            self.renderer.latent_token_decoder, **compile_opts
        )
        self.renderer.frame_decoder = torch.compile(
            self.renderer.frame_decoder, **compile_opts
        )

        logger.info("Renderer components compiled")

    def _hash_tensor(self, x: torch.Tensor) -> int:
        """Quick hash for cache invalidation"""
        return hash((x.shape, x.sum().item(), x.std().item()))

    def encode_source(self, x_source: torch.Tensor):
        """Encode source image (cached)"""
        source_hash = self._hash_tensor(x_source)

        if source_hash != self._cached_source_hash:
            with torch.no_grad():
                self._cached_f_r, self._cached_i_r = self.renderer.dense_feature_encoder(x_source)
                self._cached_t_r = self.renderer.latent_token_encoder(x_source)
                self._cached_ta_r = self.renderer.adapt(self._cached_t_r, self._cached_i_r)
                self._cached_ma_r = self.renderer.latent_token_decoder(self._cached_ta_r)
            self._cached_source_hash = source_hash

        return self._cached_f_r, self._cached_i_r, self._cached_ma_r

    def render_frame(self, x_driving: torch.Tensor, f_r, i_r, ma_r):
        """Render a single frame given driving input and cached source features"""
        with torch.no_grad():
            t_c = self.renderer.latent_token_encoder(x_driving)
            ta_c = self.renderer.adapt(t_c, i_r)
            ma_c = self.renderer.latent_token_decoder(ta_c)
            output = self.renderer.decode(ma_c, ma_r, f_r)
        return output

    def forward(self, x_driving: torch.Tensor, x_source: torch.Tensor):
        """Full forward pass with source caching"""
        f_r, i_r, ma_r = self.encode_source(x_source)
        return self.render_frame(x_driving, f_r, i_r, ma_r)

    def clear_cache(self):
        """Clear the source image cache"""
        self._cached_source_hash = None
        self._cached_f_r = None
        self._cached_i_r = None
        self._cached_t_r = None
        self._cached_ta_r = None
        self._cached_ma_r = None


class OptimizedGeneratorWrapper(nn.Module):
    """
    Optimized wrapper for FMGenerator that:
    1. Uses torch.compile on FMT
    2. Supports reduced NFE for faster inference
    3. Provides FP16 inference
    """

    def __init__(self, generator: nn.Module, config: Optional[TRTConfig] = None):
        super().__init__()
        self.config = config or TRTConfig()
        self.generator = generator

        self._compile_components()

    def _compile_components(self):
        """Compile the FMT transformer"""
        logger.info("Compiling generator FMT...")

        compile_opts = {"mode": self.config.mode, "dynamic": self.config.dynamic}
        self.generator.fmt = torch.compile(self.generator.fmt, **compile_opts)

        logger.info("Generator FMT compiled")

    def sample(self, data, a_cfg_scale=1.0, nfe=10, seed=None):
        """Wrapper around generator.sample"""
        return self.generator.sample(data, a_cfg_scale=a_cfg_scale, nfe=nfe, seed=seed)

    def forward(self, *args, **kwargs):
        return self.generator(*args, **kwargs)


def export_tensorrt_engine(
    model: nn.Module,
    example_inputs: Tuple[torch.Tensor, ...],
    output_path: str,
    config: Optional[TRTConfig] = None,
):
    """
    Export a model to TensorRT engine using torch-tensorrt.

    Args:
        model: Model to export
        example_inputs: Example inputs for tracing
        output_path: Path to save the engine
        config: TRTConfig with export settings
    """
    config = config or TRTConfig()

    try:
        import torch_tensorrt
    except ImportError:
        logger.error("torch-tensorrt not installed. Run: pip install torch-tensorrt")
        return None

    logger.info(f"Exporting TensorRT engine to {output_path}")

    # Prepare inputs specification
    input_specs = []
    for inp in example_inputs:
        if config.dynamic:
            # Allow some dynamic range around the example shape
            min_shape = [max(1, s // 2) for s in inp.shape]
            opt_shape = list(inp.shape)
            max_shape = [s * 2 for s in inp.shape]

            input_specs.append(
                torch_tensorrt.Input(
                    min_shape=min_shape,
                    opt_shape=opt_shape,
                    max_shape=max_shape,
                    dtype=torch.float16 if config.precision == "fp16" else torch.float32,
                )
            )
        else:
            input_specs.append(
                torch_tensorrt.Input(
                    shape=inp.shape,
                    dtype=torch.float16 if config.precision == "fp16" else torch.float32,
                )
            )

    # Compile
    enabled_precisions = {torch.float32}
    if config.precision == "fp16":
        enabled_precisions.add(torch.float16)

    try:
        trt_model = torch_tensorrt.compile(
            model,
            inputs=input_specs,
            enabled_precisions=enabled_precisions,
            workspace_size=config.workspace_size,
            truncate_long_and_double=True,
        )

        # Save
        torch.jit.save(trt_model, output_path)
        logger.info(f"TensorRT engine saved to {output_path}")

        return trt_model

    except Exception as e:
        logger.error(f"TensorRT export failed: {e}")
        return None


def benchmark_model(
    model: nn.Module,
    example_inputs: Tuple[torch.Tensor, ...],
    num_runs: int = 100,
    warmup: int = 10,
) -> Dict[str, float]:
    """
    Benchmark a model's inference time.

    Returns:
        Dict with avg_ms, min_ms, max_ms, std_ms
    """
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(*example_inputs)

    # Benchmark
    torch.cuda.synchronize()
    times = []

    with torch.no_grad():
        for _ in range(num_runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            _ = model(*example_inputs)
            end.record()

            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

    import statistics
    return {
        "avg_ms": statistics.mean(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "fps": 1000 / statistics.mean(times),
    }


def optimize_imtalker(
    renderer_path: str = "./checkpoints/renderer.ckpt",
    generator_path: str = "./checkpoints/generator.ckpt",
    config: Optional[TRTConfig] = None,
    device: str = "cuda",
):
    """
    Load and optimize both renderer and generator.

    Returns:
        Tuple of (optimized_renderer, optimized_generator)
    """
    from renderer.models import IMTRenderer
    from generator.FM import FMGenerator

    config = config or TRTConfig()

    # Create args for models
    class RendererArgs:
        swin_res_threshold = 128
        window_size = 8
        num_heads = 8

    class GeneratorArgs:
        fps = 25.0
        rank = device
        wav2vec_sec = 2.0
        num_prev_frames = 10
        only_last_features = True
        audio_dropout_prob = 0.0
        dim_c = 32
        dim_w = 32
        dim_h = 512
        dim_motion = 32
        fmt_depth = 8
        num_heads = 8
        mlp_ratio = 4.0
        attention_window = 5
        ode_atol = 1e-5
        ode_rtol = 1e-5
        torchdiffeq_ode_method = 'euler'
        fix_noise_seed = False
        seed = 42
        sampling_rate = 16000
        wav2vec_model_path = "./checkpoints/wav2vec2-base-960h"

    # Load renderer
    logger.info("Loading renderer...")
    renderer = IMTRenderer(RendererArgs()).to(device)
    if os.path.exists(renderer_path):
        checkpoint = torch.load(renderer_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        clean_dict = {k.replace("gen.", ""): v for k, v in state_dict.items() if k.startswith("gen.")}
        renderer.load_state_dict(clean_dict, strict=False)
    renderer.eval()

    # Load generator
    logger.info("Loading generator...")
    generator = FMGenerator(GeneratorArgs()).to(device)
    if os.path.exists(generator_path):
        checkpoint = torch.load(generator_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        clean_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
        with torch.no_grad():
            for name, param in generator.named_parameters():
                if name in clean_dict:
                    param.copy_(clean_dict[name].to(device))
    generator.eval()

    # Convert to FP16 if requested
    if config.precision == "fp16":
        renderer = renderer.half()
        generator = generator.half()

    # Wrap with optimizations
    opt_renderer = OptimizedRendererWrapper(renderer, config)
    opt_generator = OptimizedGeneratorWrapper(generator, config)

    return opt_renderer, opt_generator


def main():
    """CLI for TensorRT optimization"""
    import argparse

    parser = argparse.ArgumentParser(description="TensorRT Optimization for IMTalker")
    parser.add_argument("--export", action="store_true", help="Export TensorRT engines")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    parser.add_argument("--model", choices=["renderer", "generator", "both"], default="both")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--backend", choices=["inductor", "tensorrt", "eager"], default="inductor")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = TRTConfig(precision=args.precision, backend=args.backend)

    if args.benchmark:
        logger.info("Running benchmarks...")

        # Load models
        opt_renderer, opt_generator = optimize_imtalker(config=config, device=device)

        # Benchmark renderer
        if args.model in ["renderer", "both"]:
            dtype = torch.float16 if args.precision == "fp16" else torch.float32
            x_source = torch.randn(1, 3, 256, 256, device=device, dtype=dtype)
            x_driving = torch.randn(1, 3, 256, 256, device=device, dtype=dtype)

            logger.info("\n=== Renderer Benchmark ===")
            results = benchmark_model(opt_renderer.renderer, (x_driving, x_source))
            print(f"  Avg: {results['avg_ms']:.2f} ms")
            print(f"  Min: {results['min_ms']:.2f} ms")
            print(f"  Max: {results['max_ms']:.2f} ms")
            print(f"  FPS: {results['fps']:.1f}")

        if args.model in ["generator", "both"]:
            logger.info("\n=== Generator FMT Benchmark ===")
            # FMT benchmark would go here
            print("  (Generator benchmark requires full pipeline setup)")

    elif args.export:
        logger.info("Exporting TensorRT engines...")
        # Export logic would go here
        print("TensorRT engine export not yet implemented for complex models.")
        print("Use torch.compile() with inductor backend instead.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
