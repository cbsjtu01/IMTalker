#!/usr/bin/env python3
"""
FLOPs Profiler for IMTalker using fvcore
========================================
Profiles both the Renderer and Generator to identify computational bottlenecks.

Usage:
    python profile_flops.py
"""

import os
import sys
import torch
import torch.nn as nn
from collections import defaultdict
import time
from typing import Any, Dict, List, Tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# fvcore imports
from fvcore.nn import FlopCountAnalysis, flop_count_table, flop_count_str
from fvcore.nn.jit_handles import Handle, get_shape

from generator.FM import FMGenerator
from renderer.models import IMTRenderer


# ============================================================================
# Custom FLOPs handlers for operations fvcore doesn't cover well
# ============================================================================

def scaled_dot_product_attention_flop_jit(inputs: List[Any], outputs: List[Any]) -> int:
    """
    Count FLOPs for F.scaled_dot_product_attention
    Q @ K^T: B*H*N*N*d  (2 ops per multiply-add)
    softmax: ~5*B*H*N*N
    attn @ V: B*H*N*N*d (2 ops per multiply-add)
    """
    q_shape = get_shape(inputs[0])  # (B, H, N, d)
    if len(q_shape) != 4:
        return 0
    B, H, N, d = q_shape
    # Q@K^T + softmax + attn@V
    flops = B * H * (2 * N * N * d + 5 * N * N + 2 * N * N * d)
    return flops


def einsum_flop_jit(inputs: List[Any], outputs: List[Any]) -> int:
    """Count FLOPs for einsum operations (rough estimate)"""
    output = outputs[0]
    return output.numel() * 2


# ============================================================================
# Profiling functions
# ============================================================================

class DetailedProfiler:
    """Wrapper around fvcore with timing and memory tracking"""

    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device
        self.custom_ops = {
            "aten::scaled_dot_product_attention": scaled_dot_product_attention_flop_jit,
            "aten::einsum": einsum_flop_jit,
        }

    def profile(self, *inputs, warmup: int = 3, runs: int = 10) -> Dict:
        """Run fvcore FLOPs analysis + timing"""
        self.model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model(*inputs)

        # FLOPs analysis
        flops = FlopCountAnalysis(self.model, inputs)
        flops.set_op_handle(**self.custom_ops)

        # Timing
        torch.cuda.synchronize()
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]

        with torch.no_grad():
            for i in range(runs):
                start_events[i].record()
                _ = self.model(*inputs)
                end_events[i].record()

        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        avg_time = sum(times) / len(times)
        min_time = min(times)

        # Memory
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = self.model(*inputs)
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2

        return {
            "flops": flops,
            "total_flops": flops.total(),
            "avg_time_ms": avg_time,
            "min_time_ms": min_time,
            "peak_memory_mb": peak_mem,
        }


def profile_renderer_components(device: str = "cuda"):
    """Profile individual renderer components"""
    print("\n" + "="*100)
    print("🔍 PROFILING IMT RENDERER COMPONENTS")
    print("="*100)

    class Args:
        swin_res_threshold = 128
        window_size = 8
        num_heads = 8

    args = Args()
    renderer = IMTRenderer(args).to(device).eval()

    # Input
    x = torch.randn(1, 3, 256, 256, device=device)

    results = {}

    # Profile each component
    components = [
        ("dense_feature_encoder", renderer.dense_feature_encoder, (x,)),
        ("latent_token_encoder", renderer.latent_token_encoder, (x,)),
    ]

    for name, module, inputs in components:
        profiler = DetailedProfiler(module, device)
        result = profiler.profile(*inputs)
        results[name] = result

        print(f"\n{'─'*80}")
        print(f"📦 {name}")
        print(f"{'─'*80}")
        print(f"   Total FLOPs: {result['total_flops']/1e9:.4f} GFLOPs")
        print(f"   Avg Time:    {result['avg_time_ms']:.3f} ms")
        print(f"   Peak Memory: {result['peak_memory_mb']:.1f} MB")

    # Profile latent_token_decoder
    t_r = renderer.latent_token_encoder(x)
    _, i_r = renderer.dense_feature_encoder(x)
    ta_r = renderer.adapt(t_r, i_r)

    profiler = DetailedProfiler(renderer.latent_token_decoder, device)
    result = profiler.profile(ta_r)
    results["latent_token_decoder"] = result

    print(f"\n{'─'*80}")
    print(f"📦 latent_token_decoder (StyleConv x13)")
    print(f"{'─'*80}")
    print(f"   Total FLOPs: {result['total_flops']/1e9:.4f} GFLOPs")
    print(f"   Avg Time:    {result['avg_time_ms']:.3f} ms")
    print(f"   Peak Memory: {result['peak_memory_mb']:.1f} MB")

    return results, renderer


def profile_renderer_full(device: str = "cuda"):
    """Profile full renderer forward pass"""
    print("\n" + "="*100)
    print("🔍 PROFILING FULL RENDERER FORWARD PASS")
    print("="*100)

    class Args:
        swin_res_threshold = 128
        window_size = 8
        num_heads = 8

    args = Args()
    renderer = IMTRenderer(args).to(device).eval()

    x_source = torch.randn(1, 3, 256, 256, device=device)
    x_driving = torch.randn(1, 3, 256, 256, device=device)

    profiler = DetailedProfiler(renderer, device)
    result = profiler.profile(x_driving, x_source)

    print(f"\n{'─'*80}")
    print(f"📦 IMTRenderer Full Forward")
    print(f"{'─'*80}")
    print(f"   Total FLOPs: {result['total_flops']/1e9:.4f} GFLOPs")
    print(f"   Avg Time:    {result['avg_time_ms']:.3f} ms")
    print(f"   Min Time:    {result['min_time_ms']:.3f} ms")
    print(f"   Peak Memory: {result['peak_memory_mb']:.1f} MB")
    print(f"   Throughput:  {1000/result['avg_time_ms']:.1f} FPS")

    # Print detailed breakdown
    print(f"\n📊 FLOPs by Module:")
    print(flop_count_table(result["flops"], max_depth=4))

    return result


def profile_renderer_decode(device: str = "cuda"):
    """Profile the decode function (IMT + synthesis) - the main bottleneck"""
    print("\n" + "="*100)
    print("🔍 PROFILING RENDERER DECODE (IMT + SYNTHESIS)")
    print("="*100)

    class Args:
        swin_res_threshold = 128
        window_size = 8
        num_heads = 8

    args = Args()
    renderer = IMTRenderer(args).to(device).eval()

    x = torch.randn(1, 3, 256, 256, device=device)

    # Get intermediate representations
    with torch.no_grad():
        f_r, i_r = renderer.dense_feature_encoder(x)
        t_r = renderer.latent_token_encoder(x)
        t_c = renderer.latent_token_encoder(x)  # Same for profiling
        ta_r = renderer.adapt(t_r, i_r)
        ta_c = renderer.adapt(t_c, i_r)
        ma_r = renderer.latent_token_decoder(ta_r)
        ma_c = renderer.latent_token_decoder(ta_c)

    # Profile decode
    class DecodeWrapper(nn.Module):
        def __init__(self, renderer):
            super().__init__()
            self.renderer = renderer

        def forward(self, ma_c, ma_r, f_r):
            return self.renderer.decode(ma_c, ma_r, f_r)

    decode_wrapper = DecodeWrapper(renderer).to(device).eval()

    # fvcore needs tuple inputs
    profiler = DetailedProfiler(decode_wrapper, device)

    # Manual timing since decode takes tuple of tuples
    torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(10):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = renderer.decode(ma_c, ma_r, f_r)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

    avg_time = sum(times) / len(times)

    print(f"\n{'─'*80}")
    print(f"📦 decode() - IMT Cross-Attention + Synthesis")
    print(f"{'─'*80}")
    print(f"   Avg Time:    {avg_time:.3f} ms")
    print(f"   Throughput:  {1000/avg_time:.1f} FPS")

    # Profile individual IMT blocks
    print(f"\n   IMT Blocks by Resolution:")
    for i, (imt_block, f) in enumerate(zip(renderer.imt, f_r)):
        torch.cuda.synchronize()
        times = []
        with torch.no_grad():
            for _ in range(10):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                if imt_block.is_standard_attention:
                    _ = imt_block.coarse_stage(ma_c[i], ma_r[i], f)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))

        avg = sum(times) / len(times)
        res = f.shape[-1]
        attn_type = "Standard" if imt_block.is_standard_attention else "GuidedResampler"
        print(f"      [{i}] {res:>3}x{res:<3} ({attn_type:<16}): {avg:.3f} ms")

    return avg_time


def profile_generator(device: str = "cuda"):
    """Profile the Flow Matching Generator"""
    print("\n" + "="*100)
    print("🔍 PROFILING FM GENERATOR")
    print("="*100)

    class Args:
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

    args = Args()

    if not os.path.exists(args.wav2vec_model_path):
        print("⚠️  Wav2Vec model not found. Run app.py first to download.")
        return None

    generator = FMGenerator(args).to(device).eval()

    # Profile just the FMT (transformer) part
    num_frames = 50 + 10  # clip + prev frames
    batch_size = 1

    t = torch.tensor([0.5], device=device)
    x = torch.randn(batch_size, num_frames, args.dim_motion, device=device)
    a = torch.randn(batch_size, num_frames, args.dim_c, device=device)  # projected audio
    prev_x = torch.zeros(batch_size, 0, args.dim_c, device=device)
    prev_a = torch.zeros(batch_size, 0, args.dim_w, device=device)
    ref_x = torch.randn(batch_size, args.dim_motion, device=device)
    gaze = torch.randn(batch_size, num_frames, args.dim_c, device=device)
    prev_gaze = torch.zeros(batch_size, 0, args.dim_c, device=device)
    pose = torch.randn(batch_size, num_frames, args.dim_c, device=device)
    prev_pose = torch.zeros(batch_size, 0, args.dim_c, device=device)
    cam = torch.randn(batch_size, num_frames, args.dim_c, device=device)
    prev_cam = torch.zeros(batch_size, 0, args.dim_c, device=device)

    # Wrap FMT for profiling
    class FMTWrapper(nn.Module):
        def __init__(self, fmt):
            super().__init__()
            self.fmt = fmt

        def forward(self, t, x, a, ref_x, gaze, pose, cam):
            return self.fmt(
                t=t, x=x, a=a,
                prev_x=None, prev_a=None, ref_x=ref_x,
                gaze=gaze, prev_gaze=None,
                pose=pose, prev_pose=None,
                cam=cam, prev_cam=None,
                train=False
            )

    fmt_wrapper = FMTWrapper(generator.fmt).to(device).eval()

    # Profile FMT
    profiler = DetailedProfiler(fmt_wrapper, device)
    result = profiler.profile(t, x, a, ref_x, gaze, pose, cam)

    print(f"\n{'─'*80}")
    print(f"📦 FlowMatchingTransformer (FMT) - Single Forward")
    print(f"{'─'*80}")
    print(f"   Total FLOPs: {result['total_flops']/1e9:.4f} GFLOPs")
    print(f"   Avg Time:    {result['avg_time_ms']:.3f} ms")
    print(f"   Peak Memory: {result['peak_memory_mb']:.1f} MB")

    print(f"\n   ODE Solver Estimates (NFE = number of function evaluations):")
    for nfe in [5, 10, 20]:
        est_time = result['avg_time_ms'] * nfe
        print(f"      NFE={nfe:>2}: {est_time:.1f} ms ({1000/est_time:.1f} chunks/sec)")

    print(f"\n   With CFG (2x batch):")
    for nfe in [5, 10, 20]:
        est_time = result['avg_time_ms'] * nfe * 2
        print(f"      NFE={nfe:>2}: {est_time:.1f} ms ({1000/est_time:.1f} chunks/sec)")

    # Print breakdown
    print(f"\n📊 FMT FLOPs by Module:")
    print(flop_count_table(result["flops"], max_depth=3))

    return result


def print_summary(renderer_result, generator_result, decode_time):
    """Print final summary"""
    print("\n" + "="*100)
    print("📋 IMTALKER PERFORMANCE SUMMARY")
    print("="*100)

    r_flops = renderer_result['total_flops'] / 1e9 if renderer_result else 0
    r_time = renderer_result['avg_time_ms'] if renderer_result else 0
    g_flops = generator_result['total_flops'] / 1e9 if generator_result else 0
    g_time = generator_result['avg_time_ms'] if generator_result else 0

    print(f"""
    ┌────────────────────────────────────────────────────────────────────────┐
    │                         RENDERER (per frame)                           │
    ├────────────────────────────────────────────────────────────────────────┤
    │   Total FLOPs:      {r_flops:>10.2f} GFLOPs                                │
    │   Forward Time:     {r_time:>10.2f} ms                                     │
    │   Throughput:       {1000/r_time if r_time else 0:>10.1f} FPS                                     │
    │   decode() Time:    {decode_time:>10.2f} ms  ← MAIN BOTTLENECK               │
    ├────────────────────────────────────────────────────────────────────────┤
    │                      GENERATOR (per chunk, ~2s audio)                  │
    ├────────────────────────────────────────────────────────────────────────┤
    │   FMT FLOPs:        {g_flops:>10.2f} GFLOPs (single step)                  │
    │   FMT Time:         {g_time:>10.2f} ms (single step)                       │
    │   Full ODE (10 NFE):{g_time*10:>10.2f} ms                                     │
    │   With CFG:         {g_time*20:>10.2f} ms  ← 2x for classifier-free guidance │
    └────────────────────────────────────────────────────────────────────────┘
    """)

    # Bottleneck analysis
    print("""
    🎯 TOP BOTTLENECKS (by time contribution):
    ──────────────────────────────────────────
    1. Renderer decode()          - IMT cross-attention at 6 scales
       └─ Standard attention at low-res (8x8, 16x16, 32x32, 64x64)
       └─ GuidedResampler at high-res (128x128, 256x256)

    2. latent_token_decoder       - 13 StyleConv layers
       └─ Modulated convolutions are compute-heavy

    3. Generator FMT              - 8 transformer blocks
       └─ Attention over 60 tokens (50 frames + 10 prev)
       └─ Runs 10-20x for ODE solving

    4. dense_feature_encoder      - 6 downsampling conv blocks
       └─ Processes source image once per inference

    💡 OPTIMIZATION RECOMMENDATIONS:
    ─────────────────────────────────
    1. Use torch.compile() on renderer - can give 20-40% speedup
    2. Reduce ODE NFE steps: 10→5 for 2x generator speedup
    3. Use FP16 inference: ~2x memory savings, ~1.5x speed
    4. TensorRT for deployment: ~2-3x overall speedup
    5. Cache dense_feature_encoder output for same source image
    6. Consider FlashAttention for transformer blocks
    """)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️  Device: {device}")

    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   CUDA: {torch.version.cuda}")
        print(f"   PyTorch: {torch.__version__}")

    # Profile components
    _, renderer = profile_renderer_components(device)

    # Profile full renderer
    renderer_result = profile_renderer_full(device)

    # Profile decode specifically
    decode_time = profile_renderer_decode(device)

    # Profile generator
    generator_result = profile_generator(device)

    # Summary
    print_summary(renderer_result, generator_result, decode_time)


if __name__ == "__main__":
    main()
