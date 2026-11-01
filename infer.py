#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import tempfile

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pathlib import Path

import torch

from corridorkey.runtime import (
    InferenceConfig,
    RuntimePaths,
    WriterConfig,
    apply_weight_policy,
    configure_allocator,
    drop_unused_foundation_parts,
    load_model,
    run_inference,
)
from corridorkey.video_io import require_ffmpeg


def _parse_outputs(values):
    if not values:
        return {"alpha", "fg"}
    out = set()
    aliases = {
        "raw_fg": "fg",
        "foreground": "fg",
    }
    for value in values:
        for part in str(value).split(","):
            part = part.strip().lower()
            if part:
                part = aliases.get(part, part)
                out.add(part)
    allowed = {"alpha", "fg", "checker", "cutout"}
    bad = sorted(out - allowed)
    if bad:
        raise ValueError(f"unsupported --outputs value(s): {bad}")
    return out


def main():
    ap = argparse.ArgumentParser(
        description="CorridorKey v2 native-resolution raw-video inference")
    ap.add_argument("--input", required=True,
                    help="Input video file or directory of image frames.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--weights", default=None,
                    help="Runtime checkpoint (.safetensors or legacy .pt). "
                         "Defaults to weights/corridorkey_v2.safetensors.")
    ap.add_argument("--hint_first_frame", default=None,
                    help="Optional matte image/dir; first frame only.")
    ap.add_argument("--hint_video", default=None,
                    help="Optional full-frame hint video or frame directory.")
    ap.add_argument("--hint_quality", type=float, default=0.95)
    ap.add_argument("--carry_hint_quality", type=float, default=0.95)
    ap.add_argument("--no_carry_hint", action="store_true")
    ap.add_argument("--hann_chunk", type=int, default=None,
                    help="Frames per stage-1 window. Default: chosen "
                         "automatically from free GPU memory.")
    ap.add_argument("--hann_stride", type=int, default=None,
                    help="Frames between stage-1 windows. Default: half of "
                         "--hann_chunk.")
    ap.add_argument("--native_resolution", action="store_true", default=True,
                    help="Accepted for clarity; native resolution is always used.")
    ap.add_argument("--low_vram", action="store_true",
                    help="Staged low-memory inference. Passing this or "
                         "--hann_chunk turns off automatic selection.")
    ap.add_argument("--temp_dir",
                    default=str(Path(tempfile.gettempdir()) / "corridorkey_v2"))
    ap.add_argument("--outputs", nargs="*", default=None,
                    help=("Any of: alpha fg/raw_fg checker cutout. "
                          "Default: alpha fg."))
    ap.add_argument("--num_frames", type=int, default=-1)
    ap.add_argument("--start_mode", choices=("begin", "middle", "random_middle"),
                    default="begin")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model_scale", type=float, default=1.0,
                    help="Lanczos scale before model inference; 1.0 is native.")
    ap.add_argument("--frame_dir_fps", type=float, default=24.0)
    ap.add_argument("--frame_dir_linear", action="store_true",
                    help="Treat frame-directory inputs as scene-linear RGB.")
    ap.add_argument("--cutout_linear", action="store_true",
                    help="Export cutout as straight linear RGB instead of straight sRGB.")
    ap.add_argument("--despill_strength", type=float, default=0.5,
                    help="Despill strength from 0.0 (none) to 1.0 (full). Default: 0.5.")
    ap.add_argument("--linear_brightness", type=float, default=1.0)
    ap.add_argument("--linear_contrast", type=float, default=1.0)
    ap.add_argument("--linear_contrast_pivot", type=float, default=0.18)
    ap.add_argument("--output_format", choices=("video", "exr"),
                    default="video",
                    help=("video: ffmpeg-encoded preview videos (display-"
                          "referred, clipped to [0,1]). exr: unclamped scene-"
                          "linear OpenEXR image sequences (full HDR float)."))
    ap.add_argument("--exr_dtype", choices=("half", "float"), default="half",
                    help="EXR pixel type for --output_format exr. Default: half.")
    ap.add_argument("--ffmpeg_bin", default="ffmpeg")
    ap.add_argument("--ffmpeg_codec", default="libx264")
    ap.add_argument("--crf", type=int, default=12)
    ap.add_argument("--preset", default="medium")
    ap.add_argument("--pix_fmt", default="yuv444p10le",
                    help="Pixel format for x264 preview outputs. Default is "
                         "10-bit; pass yuv444p for 8-bit compatibility.")
    ap.add_argument("--bitrate", default=None)
    ap.add_argument("--ffmpeg_threads", type=int, default=None,
                    help="Threads per ffmpeg encoder. Default 2; with --fast, "
                         "half the CPU cores (up to 8).")
    ap.add_argument("--stage2_batch", type=int, default=1,
                    help="Number of /1 output frames to run/render per stage-2 batch.")
    ap.add_argument("--low_vram_foundation_chunk", type=int, default=4)
    ap.add_argument("--low_vram_mlp_chunk_tokens", type=int, default=65536)
    ap.add_argument("--low_vram_enc_frame_chunk", type=int, default=1)
    ap.add_argument("--low_vram_swin_window_batch", type=int, default=16)
    ap.add_argument("--low_vram_full_attn_query_chunk_tokens", type=int,
                    default=2048)
    ap.add_argument("--frozen_weights", action="store_true",
                    help="Mark all parameters requires_grad=False so autocast "
                         "does not cache bf16 copies of the weights. Same output, "
                         "less VRAM.")
    ap.add_argument("--bf16_weights", action="store_true",
                    help="Store Linear/Conv weights as bf16 (implies "
                         "--frozen_weights). Same output, about half the weight "
                         "memory on GPU and in RAM.")
    ap.add_argument("--lean_activations", action="store_true",
                    help="Batched-window attention without full-grid padded "
                         "copies. Same output, less VRAM per frame.")
    ap.add_argument("--prefetch_frames", action="store_true",
                    help="Decode and preprocess the next window's frames on a "
                         "background thread while the GPU works. Same output, "
                         "faster.")
    ap.add_argument("--cache_region_indices", action="store_true",
                    help="Keep shifted-window attention index tensors on the "
                         "GPU instead of uploading them on every call (each "
                         "upload stalls the GPU). Same output, faster on "
                         "slower CPUs.")
    ap.add_argument("--activation_arena", action="store_true",
                    help="Without expandable allocator segments (Windows), "
                         "reserve each run's estimated working memory as one "
                         "block before stage 1, so freed memory merges back "
                         "instead of stranding in pieces. Same output, less "
                         "reserved VRAM. Sized from the optimized path's "
                         "memory table, so meant for use with the options "
                         "above.")
    ap.add_argument("--pinned_io", action="store_true",
                    help="Quantize output frames on the GPU and copy them "
                         "through pinned host memory while the GPU starts on "
                         "the next frame, instead of stopping it for each "
                         "copy. Same output, faster when the GPU outruns the "
                         "host.")
    ap.add_argument("--async_baton", action="store_true",
                    help="In low-VRAM mode, keep stage 1's windows back to "
                         "back on the GPU: the carried hint stays on the GPU, "
                         "and each window's output is copied back and blended "
                         "into the finished frames while the GPU runs the "
                         "next window. Same output.")
    ap.add_argument("--lean_reads", action="store_true",
                    help="Where a stage only needs the model's view of each "
                         "frame (low-VRAM stages 1 and 2), skip preparing the "
                         "display views. Same output, less CPU time per "
                         "frame (most at 4K and with linear EXR input).")
    ap.add_argument("--read_ahead", action="store_true",
                    help="With --prefetch_frames: read stage 1's first window "
                         "while the prepass runs, later windows on worker "
                         "threads, and stage 2's frames eight at a time. Same "
                         "output, less waiting for frames on fast GPUs.")
    ap.add_argument("--decode_once", action="store_true",
                    help="With --prefetch_frames, decode each frame once: the "
                         "prepass also writes the model's input (its exact "
                         "bits; in normal mode the foundation models' input "
                         "too) to temporary files that the later stages read "
                         "instead of decoding the clip again. Same output; "
                         "needs about 13 MB per 1080p frame and 53 MB per 4K "
                         "frame in --temp_dir (twice that in normal mode).")
    ap.add_argument("--gpu_prep", action="store_true",
                    help="With --decode_once and a frame folder, convert each "
                         "decoded frame (sRGB/linear and the model's asinh "
                         "curve) on the GPU instead of the CPU threads. Same "
                         "output: for 8-bit images and half-float EXRs the GPU "
                         "gives the CPU's bits for every possible value; "
                         "float EXRs, where a few "
                         "values round differently, stay on the CPU. Faster "
                         "where frame preparation sets the pace (4K, EXR).")
    ap.add_argument("--read_workers", type=int, default=None,
                    help="Threads the prepass prepares frames on (default 4; "
                         "with --optimized, half the CPU threads, 4 to 12).")
    ap.add_argument("--fast_load", action="store_true",
                    help="Build the model without first filling its weights "
                         "with random values (every one is then loaded from "
                         "the checkpoints). Same weights, faster start.")
    ap.add_argument("--lean_foundations", action="store_true",
                    help="Free the parts of the MoGe model that inference "
                         "never runs (its neck and four heads; only its "
                         "encoder is used). Same output, 65 MiB less GPU "
                         "memory (32 MiB with --bf16_weights).")
    ap.add_argument("--exr_workers", type=int, default=None,
                    help="Threads per EXR output that compress and write "
                         "frames at once (default 1; --optimized and --fast: "
                         "a quarter of the CPU cores, 2 to 8). Each frame's "
                         "file is still written whole by one thread, so the "
                         "files are the same; ZIP-compressing a 4K frame "
                         "takes a core seconds.")
    ap.add_argument("--fast", action="store_true",
                    help="--optimized plus more threads per video encoder "
                         "(half the CPU cores, up to 8). The keyed frames are "
                         "identical; the encoded files differ slightly, since "
                         "x264's choices depend on its thread count. On fast "
                         "GPUs the encoder is often what sets the pace.")
    ap.add_argument("--optimized", action="store_true",
                    help="The same-output options above that measured "
                         "faster (--bf16_weights --lean_activations "
                         "--prefetch_frames --cache_region_indices "
                         "--activation_arena --pinned_io --lean_reads "
                         "--read_ahead --fast_load --decode_once "
                         "--lean_foundations --gpu_prep --async_baton, "
                         "--exr_workers), expandable "
                         "allocator segments on Linux, and auto window sizes "
                         "from the optimized path's memory table. Same output "
                         "for the same window; auto mode picks larger windows "
                         "on the same GPU.")
    args = ap.parse_args()
    if args.fast:
        args.optimized = True
    if args.ffmpeg_threads is None:
        args.ffmpeg_threads = (min(8, max(2, (os.cpu_count() or 4) // 2))
                               if args.fast else 2)
    if args.read_workers is None:
        args.read_workers = (min(12, max(4, (os.cpu_count() or 8) // 2))
                             if args.optimized or args.fast else 4)
    if args.optimized:
        args.bf16_weights = args.lean_activations = args.prefetch_frames = True
        args.cache_region_indices = args.activation_arena = True
        args.pinned_io = args.lean_reads = args.read_ahead = True
        args.fast_load = args.decode_once = args.lean_foundations = True
        args.gpu_prep = args.async_baton = True
    if args.exr_workers is None:
        args.exr_workers = (min(8, max(2, (os.cpu_count() or 8) // 4))
                            if args.optimized else 1)

    if not torch.cuda.is_available():
        raise RuntimeError("CorridorKey v2 runtime requires a CUDA GPU.")
    if args.optimized:
        configure_allocator()
    require_ffmpeg(args.ffmpeg_bin)
    package_root = Path(__file__).resolve().parent
    paths = RuntimePaths.from_package(package_root, args.weights)
    writer_cfg = WriterConfig(
        ffmpeg_bin=args.ffmpeg_bin,
        codec=args.ffmpeg_codec,
        crf=args.crf,
        preset=args.preset,
        pix_fmt=args.pix_fmt,
        bitrate=args.bitrate,
        threads=args.ffmpeg_threads,
    )
    # With no explicit memory settings, pick them per clip from free GPU memory.
    auto_vram = args.hann_chunk is None and args.hann_stride is None and not args.low_vram
    hann_chunk = args.hann_chunk if args.hann_chunk is not None else 80
    hann_stride = args.hann_stride if args.hann_stride is not None else max(1, hann_chunk // 2)
    infer_cfg = InferenceConfig(
        hann_chunk=hann_chunk,
        hann_stride=hann_stride,
        auto_vram=auto_vram,
        auto_vram_optimized=args.optimized,
        carry_hint=not args.no_carry_hint,
        hint_quality=args.hint_quality,
        carry_hint_quality=args.carry_hint_quality,
        model_scale=args.model_scale,
        num_frames=args.num_frames,
        start_mode=args.start_mode,
        seed=args.seed,
        low_vram=args.low_vram,
        low_vram_foundation_chunk=args.low_vram_foundation_chunk,
        low_vram_mlp_chunk_tokens=args.low_vram_mlp_chunk_tokens,
        low_vram_enc_frame_chunk=args.low_vram_enc_frame_chunk,
        low_vram_swin_window_batch=args.low_vram_swin_window_batch,
        low_vram_full_attn_query_chunk_tokens=(
            args.low_vram_full_attn_query_chunk_tokens),
        stage2_batch=args.stage2_batch,
        outputs=_parse_outputs(args.outputs),
        temp_dir=Path(args.temp_dir),
        linear_brightness=args.linear_brightness,
        linear_contrast=args.linear_contrast,
        linear_contrast_pivot=args.linear_contrast_pivot,
        frame_dir_fps=args.frame_dir_fps,
        frame_dir_linear=args.frame_dir_linear,
        cutout_linear=args.cutout_linear,
        despill_strength=args.despill_strength,
        output_format=args.output_format,
        exr_dtype=args.exr_dtype,
        lean_activations=args.lean_activations,
        prefetch_frames=args.prefetch_frames,
        cache_region_indices=args.cache_region_indices,
        activation_arena=args.activation_arena,
        pinned_io=args.pinned_io,
        async_baton=args.async_baton,
        lean_reads=args.lean_reads,
        read_ahead=args.read_ahead,
        decode_once=args.decode_once,
        read_workers=args.read_workers,
        gpu_prep=args.gpu_prep,
        exr_workers=args.exr_workers,
    )
    model = load_model(paths, device="cuda", skip_init=args.fast_load)
    apply_weight_policy(model, freeze=args.frozen_weights,
                        bf16_linear_conv=args.bf16_weights)
    if args.lean_foundations:
        drop_unused_foundation_parts(model)
    run_inference(
        model,
        args.input,
        args.output_dir,
        infer_cfg,
        writer_cfg,
        initial_hint_path=args.hint_first_frame,
        hint_video_path=args.hint_video,
    )


if __name__ == "__main__":
    main()
