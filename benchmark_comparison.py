#!/usr/bin/env python3
"""
Benchmark Comparison: Vanilla vs Optimized IMTalker
====================================================
Compares performance between:
- Vanilla PyTorch (FP32, no compile)
- Optimized (FP16, torch.compile)
"""

import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
import statistics

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from renderer.models import IMTRenderer
from generator.FM import FMGenerator


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark"""
    renderer_path: str = "./checkpoints/renderer.ckpt"
    generator_path: str = "./checkpoints/generator.ckpt"
    wav2vec_model_path: str = "./checkpoints/wav2vec2-base-960h"
    device: str = "cuda"
    warmup_runs: int = 5
    benchmark_runs: int = 20
    batch_size: int = 1
    input_size: int = 256

    # Model args
    swin_res_threshold: int = 128
    window_size: int = 8
    num_heads: int = 8
    fps: float = 25.0
    rank: str = "cuda"
    wav2vec_sec: float = 2.0
    num_prev_frames: int = 10
    only_last_features: bool = True
    audio_dropout_prob: float = 0.0
    dim_c: int = 32
    dim_w: int = 32
    dim_h: int = 512
    dim_motion: int = 32
    fmt_depth: int = 8
    mlp_ratio: float = 4.0
    attention_window: int = 5
    ode_atol: float = 1e-5
    ode_rtol: float = 1e-5
    torchdiffeq_ode_method: str = 'euler'
    fix_noise_seed: bool = False
    seed: int = 42
    sampling_rate: int = 16000


def load_renderer(cfg: BenchmarkConfig, dtype=torch.float32):
    """Load renderer model"""
    renderer = IMTRenderer(cfg).to(cfg.device)
    if os.path.exists(cfg.renderer_path):
        checkpoint = torch.load(cfg.renderer_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        clean_dict = {k.replace("gen.", ""): v for k, v in state_dict.items() if k.startswith("gen.")}
        renderer.load_state_dict(clean_dict, strict=False)

    if dtype == torch.float16:
        renderer = renderer.half()

    renderer.eval()
    return renderer


def load_generator(cfg: BenchmarkConfig, dtype=torch.float32):
    """Load generator model"""
    generator = FMGenerator(cfg).to(cfg.device)
    if os.path.exists(cfg.generator_path):
        checkpoint = torch.load(cfg.generator_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        clean_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
        with torch.no_grad():
            for name, param in generator.named_parameters():
                if name in clean_dict:
                    param.copy_(clean_dict[name].to(cfg.device))

    if dtype == torch.float16:
        generator = generator.half()

    generator.eval()
    return generator


def benchmark_renderer(
    renderer: nn.Module,
    dtype: torch.dtype,
    cfg: BenchmarkConfig,
    label: str = "Renderer"
) -> Dict[str, float]:
    """Benchmark renderer forward pass"""
    x = torch.randn(cfg.batch_size, 3, cfg.input_size, cfg.input_size,
                    device=cfg.device, dtype=dtype)

    # Warmup
    print(f"  Warming up {label}...")
    with torch.no_grad():
        for _ in range(cfg.warmup_runs):
            _ = renderer(x, x)
    torch.cuda.synchronize()

    # Benchmark
    print(f"  Benchmarking {label}...")
    times = []
    with torch.no_grad():
        for _ in range(cfg.benchmark_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = renderer(x, x)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

    return {
        "avg_ms": statistics.mean(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "fps": 1000 / statistics.mean(times),
    }


def benchmark_renderer_decode(
    renderer: nn.Module,
    dtype: torch.dtype,
    cfg: BenchmarkConfig,
    label: str = "decode()"
) -> Dict[str, float]:
    """Benchmark just the decode() function (the main bottleneck)"""
    x = torch.randn(cfg.batch_size, 3, cfg.input_size, cfg.input_size,
                    device=cfg.device, dtype=dtype)

    # Get intermediate features
    with torch.no_grad():
        f_r, i_r = renderer.dense_feature_encoder(x)
        t_r = renderer.latent_token_encoder(x)
        ta_r = renderer.adapt(t_r, i_r)
        ma_r = renderer.latent_token_decoder(ta_r)

        t_c = renderer.latent_token_encoder(x)
        ta_c = renderer.adapt(t_c, i_r)
        ma_c = renderer.latent_token_decoder(ta_c)

    # Warmup
    print(f"  Warming up {label}...")
    with torch.no_grad():
        for _ in range(cfg.warmup_runs):
            _ = renderer.decode(ma_c, ma_r, f_r)
    torch.cuda.synchronize()

    # Benchmark
    print(f"  Benchmarking {label}...")
    times = []
    with torch.no_grad():
        for _ in range(cfg.benchmark_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = renderer.decode(ma_c, ma_r, f_r)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

    return {
        "avg_ms": statistics.mean(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "fps": 1000 / statistics.mean(times),
    }


def compile_renderer(renderer: nn.Module, mode: str = "default"):
    """Apply torch.compile to renderer hot paths

    Note: Using 'default' mode instead of 'reduce-overhead' because
    the renderer has tensor reuse patterns that conflict with CUDA graphs.
    """
    renderer.latent_token_encoder = torch.compile(
        renderer.latent_token_encoder, mode=mode, dynamic=True
    )
    renderer.latent_token_decoder = torch.compile(
        renderer.latent_token_decoder, mode=mode, dynamic=True
    )
    renderer.frame_decoder = torch.compile(
        renderer.frame_decoder, mode=mode, dynamic=True
    )
    return renderer


def main():
    print("=" * 70)
    print("IMTalker Benchmark: Vanilla vs Optimized")
    print("=" * 70)

    cfg = BenchmarkConfig()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    print(f"\nDevice: {torch.cuda.get_device_name()}")
    print(f"Warmup runs: {cfg.warmup_runs}")
    print(f"Benchmark runs: {cfg.benchmark_runs}")
    print(f"Input shape: ({cfg.batch_size}, 3, {cfg.input_size}, {cfg.input_size})")

    results = {}

    # ==========================================
    # 1. Vanilla FP32 (no compile)
    # ==========================================
    print("\n" + "=" * 70)
    print("1. VANILLA FP32 (no torch.compile)")
    print("=" * 70)

    renderer_fp32 = load_renderer(cfg, dtype=torch.float32)

    results["vanilla_fp32_full"] = benchmark_renderer(
        renderer_fp32, torch.float32, cfg, "Renderer Full FP32"
    )
    results["vanilla_fp32_decode"] = benchmark_renderer_decode(
        renderer_fp32, torch.float32, cfg, "decode() FP32"
    )

    del renderer_fp32
    torch.cuda.empty_cache()

    # ==========================================
    # 2. FP16 (no compile)
    # ==========================================
    print("\n" + "=" * 70)
    print("2. FP16 (no torch.compile)")
    print("=" * 70)

    renderer_fp16 = load_renderer(cfg, dtype=torch.float16)

    results["fp16_full"] = benchmark_renderer(
        renderer_fp16, torch.float16, cfg, "Renderer Full FP16"
    )
    results["fp16_decode"] = benchmark_renderer_decode(
        renderer_fp16, torch.float16, cfg, "decode() FP16"
    )

    del renderer_fp16
    torch.cuda.empty_cache()

    # ==========================================
    # 3. FP16 + torch.compile
    # ==========================================
    print("\n" + "=" * 70)
    print("3. FP16 + torch.compile (default mode)")
    print("=" * 70)

    renderer_compiled = load_renderer(cfg, dtype=torch.float16)
    renderer_compiled = compile_renderer(renderer_compiled)

    results["compiled_fp16_full"] = benchmark_renderer(
        renderer_compiled, torch.float16, cfg, "Compiled Renderer FP16"
    )
    results["compiled_fp16_decode"] = benchmark_renderer_decode(
        renderer_compiled, torch.float16, cfg, "Compiled decode() FP16"
    )

    del renderer_compiled
    torch.cuda.empty_cache()

    # ==========================================
    # Summary
    # ==========================================
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                     RENDERER FULL FORWARD                            │")
    print("├──────────────────────────┬────────────┬─────────┬───────────────────┤")
    print("│ Configuration            │  Avg (ms)  │   FPS   │ Speedup vs FP32   │")
    print("├──────────────────────────┼────────────┼─────────┼───────────────────┤")

    baseline = results["vanilla_fp32_full"]["avg_ms"]
    for name, key in [
        ("Vanilla FP32", "vanilla_fp32_full"),
        ("FP16 only", "fp16_full"),
        ("FP16 + compile", "compiled_fp16_full"),
    ]:
        r = results[key]
        speedup = baseline / r["avg_ms"]
        print(f"│ {name:<24} │ {r['avg_ms']:>10.2f} │ {r['fps']:>7.1f} │ {speedup:>17.2f}x │")

    print("└──────────────────────────┴────────────┴─────────┴───────────────────┘")

    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                       DECODE() ONLY                                  │")
    print("├──────────────────────────┬────────────┬─────────┬───────────────────┤")
    print("│ Configuration            │  Avg (ms)  │   FPS   │ Speedup vs FP32   │")
    print("├──────────────────────────┼────────────┼─────────┼───────────────────┤")

    baseline = results["vanilla_fp32_decode"]["avg_ms"]
    for name, key in [
        ("Vanilla FP32", "vanilla_fp32_decode"),
        ("FP16 only", "fp16_decode"),
        ("FP16 + compile", "compiled_fp16_decode"),
    ]:
        r = results[key]
        speedup = baseline / r["avg_ms"]
        print(f"│ {name:<24} │ {r['avg_ms']:>10.2f} │ {r['fps']:>7.1f} │ {speedup:>17.2f}x │")

    print("└──────────────────────────┴────────────┴─────────┴───────────────────┘")

    # Calculate overall improvement
    total_speedup = results["vanilla_fp32_full"]["avg_ms"] / results["compiled_fp16_full"]["avg_ms"]

    print(f"\n🚀 Overall Speedup (FP16 + compile vs FP32): {total_speedup:.2f}x")
    print(f"   Vanilla FP32: {results['vanilla_fp32_full']['fps']:.1f} FPS")
    print(f"   Optimized:    {results['compiled_fp16_full']['fps']:.1f} FPS")


if __name__ == "__main__":
    main()
