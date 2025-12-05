#!/usr/bin/env python3
"""
Export IMTalker Models to Optimized Formats
============================================

This script exports the renderer and generator models to various optimized formats:
1. TorchScript (JIT) - Portable, no Python dependency at runtime
2. torch.compile cache - Pre-warm the compilation cache for faster startup
3. TensorRT engines - Maximum performance (when model supports it)
4. ONNX - For cross-platform deployment

Usage:
    # Export all formats
    python export_engines.py --all

    # Export specific format
    python export_engines.py --torchscript
    python export_engines.py --compile-cache
    python export_engines.py --tensorrt
    python export_engines.py --onnx

    # Benchmark exported models
    python export_engines.py --benchmark
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Constants
CHECKPOINT_DIR = "./checkpoints"
EXPORT_DIR = "./checkpoints/exported"
INPUT_SIZE = 256


class ModelConfig:
    """Configuration matching the model requirements"""
    def __init__(self):
        self.device = "cuda"
        self.swin_res_threshold = 128
        self.window_size = 8
        self.num_heads = 8
        self.fps = 25.0
        self.rank = "cuda"
        self.wav2vec_sec = 2.0
        self.num_prev_frames = 10
        self.only_last_features = True
        self.audio_dropout_prob = 0.0
        self.dim_c = 32
        self.dim_w = 32
        self.dim_h = 512
        self.dim_motion = 32
        self.fmt_depth = 8
        self.mlp_ratio = 4.0
        self.attention_window = 5
        self.ode_atol = 1e-5
        self.ode_rtol = 1e-5
        self.torchdiffeq_ode_method = 'euler'
        self.fix_noise_seed = False
        self.seed = 42
        self.sampling_rate = 16000
        self.wav2vec_model_path = "./checkpoints/wav2vec2-base-960h"


def load_renderer(config: ModelConfig) -> nn.Module:
    """Load the IMTRenderer model"""
    from renderer.models import IMTRenderer

    renderer = IMTRenderer(config).to(config.device)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "renderer.ckpt")

    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        clean_dict = {k.replace("gen.", ""): v for k, v in state_dict.items() if k.startswith("gen.")}
        renderer.load_state_dict(clean_dict, strict=False)
        logger.info(f"Loaded renderer from {ckpt_path}")

    renderer.eval()
    return renderer


def load_generator(config: ModelConfig) -> nn.Module:
    """Load the FMGenerator model"""
    from generator.FM import FMGenerator

    generator = FMGenerator(config).to(config.device)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "generator.ckpt")

    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        clean_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
        with torch.no_grad():
            for name, param in generator.named_parameters():
                if name in clean_dict:
                    param.copy_(clean_dict[name].to(config.device))
        logger.info(f"Loaded generator from {ckpt_path}")

    generator.eval()
    return generator


def get_component_inputs(renderer: nn.Module, config: ModelConfig) -> Dict[str, Tuple]:
    """
    Generate proper example inputs for each renderer component by tracing data flow.

    Each component expects different input shapes:
    - dense_feature_encoder: (B, 3, 256, 256) -> (f_r, i_r)
    - latent_token_encoder: (B, 3, 256, 256) -> t_r
    - latent_token_decoder: adapted tokens -> motion features
    - frame_decoder: motion features -> decoded frame
    """
    device = config.device
    x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)

    with torch.no_grad():
        # Get intermediate tensors by running through the model
        f_r, i_r = renderer.dense_feature_encoder(x)
        t_r = renderer.latent_token_encoder(x)
        ta_r = renderer.adapt(t_r, i_r)
        ma_r = renderer.latent_token_decoder(ta_r)

    return {
        "dense_feature_encoder": (x,),
        "latent_token_encoder": (x,),
        "latent_token_decoder": (ta_r,),  # Takes adapted tokens
        "frame_decoder": (ma_r,),  # Takes motion features
    }


def export_torchscript(renderer: nn.Module, config: ModelConfig, output_dir: str) -> bool:
    """
    Export renderer to TorchScript format.

    TorchScript allows running models without Python runtime.
    Note: Full model tracing may fail due to dynamic control flow.
    We export individual components instead.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = config.device

    logger.info("Exporting to TorchScript...")

    # Get proper inputs for each component
    component_inputs = get_component_inputs(renderer, config)

    success_count = 0
    components = [
        ("dense_feature_encoder", renderer.dense_feature_encoder),
        ("latent_token_encoder", renderer.latent_token_encoder),
        ("latent_token_decoder", renderer.latent_token_decoder),
        ("frame_decoder", renderer.frame_decoder),
    ]

    for name, module in components:
        try:
            inputs = component_inputs[name]
            # Use torch.jit.trace for these modules
            scripted = torch.jit.trace(module, inputs, strict=False)
            output_path = os.path.join(output_dir, f"{name}.pt")
            torch.jit.save(scripted, output_path)
            logger.info(f"  Exported {name} -> {output_path}")
            success_count += 1
        except Exception as e:
            logger.warning(f"  Failed to export {name}: {e}")

    logger.info(f"TorchScript export: {success_count}/{len(components)} components")
    return success_count > 0


def export_compile_cache(renderer: nn.Module, config: ModelConfig, output_dir: str) -> bool:
    """
    Pre-warm torch.compile cache.

    This runs torch.compile and saves the compiled artifacts to a cache directory.
    On next load, compilation will be much faster.
    """
    os.makedirs(output_dir, exist_ok=True)
    cache_dir = os.path.join(output_dir, "torch_compile_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Set cache directory
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir

    logger.info(f"Pre-warming torch.compile cache -> {cache_dir}")

    # Get proper inputs for each component
    component_inputs = get_component_inputs(renderer, config)

    compile_opts = {"mode": "default", "dynamic": True}

    # Compile and warm up each component
    components = [
        ("latent_token_encoder", renderer.latent_token_encoder),
        ("latent_token_decoder", renderer.latent_token_decoder),
        ("frame_decoder", renderer.frame_decoder),
    ]

    for name, module in components:
        logger.info(f"  Compiling {name}...")
        try:
            compiled = torch.compile(module, **compile_opts)
            inputs = component_inputs[name]
            # Warmup run to trigger compilation
            with torch.no_grad():
                for _ in range(3):
                    _ = compiled(*inputs)
            logger.info(f"  {name} compiled and cached")
        except Exception as e:
            logger.warning(f"  Failed to compile {name}: {e}")

    logger.info(f"Compile cache saved to: {cache_dir}")
    return True


def export_tensorrt(renderer: nn.Module, config: ModelConfig, output_dir: str) -> bool:
    """
    Export to TensorRT .engine format.

    This provides maximum inference performance but requires:
    - Static or bounded dynamic shapes
    - Operations supported by TensorRT

    Exports both:
    1. .engine files (native TensorRT format, fastest)
    2. .ts files (TorchScript with embedded TRT, easier to use)
    """
    try:
        import torch_tensorrt
    except ImportError:
        logger.error("torch_tensorrt not installed. Run: pip install torch-tensorrt")
        return False

    os.makedirs(output_dir, exist_ok=True)

    logger.info("Exporting to TensorRT...")

    # Get proper inputs for each component
    component_inputs = get_component_inputs(renderer, config)

    # Components to export with their input specs
    components = [
        ("dense_feature_encoder", renderer.dense_feature_encoder, [1, 3, INPUT_SIZE, INPUT_SIZE]),
        ("latent_token_encoder", renderer.latent_token_encoder, [1, 3, INPUT_SIZE, INPUT_SIZE]),
        ("latent_token_decoder", renderer.latent_token_decoder, None),  # Dynamic shape from adapt()
    ]

    success_count = 0

    for name, module, input_shape in components:
        try:
            logger.info(f"  Converting {name} to TensorRT...")

            # Get the actual input tensor for this component
            example_input = component_inputs[name][0]
            actual_shape = list(example_input.shape)

            # Create TensorRT input spec
            trt_input = torch_tensorrt.Input(
                min_shape=actual_shape,
                opt_shape=actual_shape,
                max_shape=actual_shape,
                dtype=torch.float32,
            )

            # Compile with dynamo (more compatible with modern PyTorch)
            trt_module = torch_tensorrt.compile(
                module,
                ir="dynamo",
                inputs=[trt_input],
                enabled_precisions={torch.float32},
                truncate_double=True,
                device=torch_tensorrt.Device(gpu_id=0),
            )

            # Save as TorchScript with embedded TRT engine
            ts_path = os.path.join(output_dir, f"{name}_trt.ts")
            torch_tensorrt.save(trt_module, ts_path, inputs=[example_input])
            logger.info(f"  Exported {name} -> {ts_path}")

            # Also export native .engine file
            engine_path = os.path.join(output_dir, f"{name}.engine")
            torch_tensorrt.save(trt_module, engine_path, output_format="torchtrt")
            logger.info(f"  Exported {name} -> {engine_path}")

            success_count += 1

        except Exception as e:
            logger.warning(f"  Failed to export {name}: {e}")

    logger.info(f"TensorRT export: {success_count}/{len(components)} components")
    return success_count > 0


def export_onnx(renderer: nn.Module, config: ModelConfig, output_dir: str) -> bool:
    """
    Export to ONNX format.

    ONNX provides cross-platform compatibility and can be optimized
    with various runtimes (ONNX Runtime, TensorRT, OpenVINO, etc.)
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Exporting to ONNX...")

    # Get proper inputs for each component
    component_inputs = get_component_inputs(renderer, config)

    components = [
        ("dense_feature_encoder", renderer.dense_feature_encoder),
        ("latent_token_encoder", renderer.latent_token_encoder),
        ("latent_token_decoder", renderer.latent_token_decoder),
        ("frame_decoder", renderer.frame_decoder),
    ]

    success_count = 0
    for name, module in components:
        output_path = os.path.join(output_dir, f"{name}.onnx")
        try:
            inputs = component_inputs[name]
            logger.info(f"  Exporting {name} (input shape: {inputs[0].shape})...")
            torch.onnx.export(
                module,
                inputs,
                output_path,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                opset_version=17,
                do_constant_folding=True,
            )
            logger.info(f"  Exported {name} -> {output_path}")
            success_count += 1
        except Exception as e:
            logger.warning(f"  Failed to export {name}: {e}")

    logger.info(f"ONNX export: {success_count}/{len(components)} components")
    return success_count > 0


def benchmark_exports(config: ModelConfig, export_dir: str) -> Dict[str, float]:
    """Benchmark different exported formats"""
    import statistics

    results = {}
    device = config.device
    x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)

    warmup = 5
    runs = 20

    # Benchmark original
    logger.info("Benchmarking original model...")
    renderer = load_renderer(config)

    with torch.no_grad():
        for _ in range(warmup):
            _ = renderer(x, x)
        torch.cuda.synchronize()

        times = []
        for _ in range(runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = renderer(x, x)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

    results["original"] = statistics.mean(times)
    logger.info(f"  Original: {results['original']:.2f} ms")

    # Benchmark torch.compile
    logger.info("Benchmarking torch.compile...")
    compile_opts = {"mode": "default", "dynamic": True}
    renderer.latent_token_encoder = torch.compile(renderer.latent_token_encoder, **compile_opts)
    renderer.latent_token_decoder = torch.compile(renderer.latent_token_decoder, **compile_opts)
    renderer.frame_decoder = torch.compile(renderer.frame_decoder, **compile_opts)

    with torch.no_grad():
        for _ in range(warmup + 3):  # Extra warmup for compilation
            _ = renderer(x, x)
        torch.cuda.synchronize()

        times = []
        for _ in range(runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = renderer(x, x)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

    results["torch_compile"] = statistics.mean(times)
    logger.info(f"  torch.compile: {results['torch_compile']:.2f} ms")

    # Benchmark TensorRT if engines exist
    trt_dir = os.path.join(export_dir, "tensorrt")
    trt_encoder_path = os.path.join(trt_dir, "latent_token_encoder_trt.ts")
    trt_decoder_path = os.path.join(trt_dir, "latent_token_decoder_trt.ts")

    if os.path.exists(trt_encoder_path) and os.path.exists(trt_decoder_path):
        logger.info("Benchmarking TensorRT engines...")

        # Load fresh renderer and replace components with TRT versions
        renderer_trt = load_renderer(config)

        try:
            # Import torch_tensorrt to register custom ops for loading
            import torch_tensorrt

            # Load TRT models - they're ExportedProgram format from dynamo
            trt_encoder_ep = torch_tensorrt.load(trt_encoder_path)
            trt_decoder_ep = torch_tensorrt.load(trt_decoder_path)

            # Convert ExportedProgram to callable module
            trt_encoder = trt_encoder_ep.module().to(device)
            trt_decoder = trt_decoder_ep.module().to(device)

            # Replace the components
            renderer_trt.latent_token_encoder = trt_encoder
            renderer_trt.latent_token_decoder = trt_decoder

            with torch.no_grad():
                for _ in range(warmup):
                    _ = renderer_trt(x, x)
                torch.cuda.synchronize()

                times = []
                for _ in range(runs):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    _ = renderer_trt(x, x)
                    torch.cuda.synchronize()
                    times.append((time.perf_counter() - start) * 1000)

            results["tensorrt"] = statistics.mean(times)
            logger.info(f"  TensorRT: {results['tensorrt']:.2f} ms")
        except Exception as e:
            logger.warning(f"  TensorRT benchmark failed: {e}")
    else:
        logger.info("TensorRT engines not found, skipping TRT benchmark")
        logger.info(f"  Run: python export_engines.py --tensorrt")

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 50)
    for name, time_ms in results.items():
        speedup = results["original"] / time_ms if name != "original" else 1.0
        fps = 1000.0 / time_ms
        logger.info(f"  {name:20s}: {time_ms:8.2f} ms  ({speedup:.2f}x)  {fps:.1f} FPS")

    return results


def main():
    parser = argparse.ArgumentParser(description="Export IMTalker models to optimized formats")
    parser.add_argument("--all", action="store_true", help="Export all formats")
    parser.add_argument("--torchscript", action="store_true", help="Export to TorchScript")
    parser.add_argument("--compile-cache", action="store_true", help="Pre-warm torch.compile cache")
    parser.add_argument("--tensorrt", action="store_true", help="Export to TensorRT")
    parser.add_argument("--onnx", action="store_true", help="Export to ONNX")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark exported models")
    parser.add_argument("--output-dir", type=str, default=EXPORT_DIR, help="Output directory")

    args = parser.parse_args()

    if not any([args.all, args.torchscript, args.compile_cache, args.tensorrt, args.onnx, args.benchmark]):
        parser.print_help()
        return

    # Check CUDA
    if not torch.cuda.is_available():
        logger.error("CUDA not available")
        return

    logger.info(f"Device: {torch.cuda.get_device_name()}")
    logger.info(f"Output directory: {args.output_dir}")

    config = ModelConfig()
    renderer = load_renderer(config)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.all or args.torchscript:
        export_torchscript(renderer, config, os.path.join(args.output_dir, "torchscript"))

    if args.all or args.compile_cache:
        export_compile_cache(renderer, config, args.output_dir)

    if args.all or args.tensorrt:
        export_tensorrt(renderer, config, os.path.join(args.output_dir, "tensorrt"))

    if args.all or args.onnx:
        export_onnx(renderer, config, os.path.join(args.output_dir, "onnx"))

    if args.benchmark:
        benchmark_exports(config, args.output_dir)

    logger.info("\nExport complete!")
    logger.info(f"Files saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
