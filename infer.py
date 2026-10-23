#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import tempfile

# Before torch loads: avoids allocator fragmentation that defeats VRAM auto-sizing.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pathlib import Path

import torch

from corridorkey.runtime import (
    InferenceConfig,
    RuntimePaths,
    WriterConfig,
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
    ap.add_argument("--ffmpeg_threads", type=int, default=2)
    ap.add_argument("--stage2_batch", type=int, default=1,
                    help="Number of /1 output frames to run/render per stage-2 batch.")
    ap.add_argument("--low_vram_foundation_chunk", type=int, default=4)
    ap.add_argument("--low_vram_mlp_chunk_tokens", type=int, default=65536)
    ap.add_argument("--low_vram_enc_frame_chunk", type=int, default=1)
    ap.add_argument("--low_vram_swin_window_batch", type=int, default=16)
    ap.add_argument("--low_vram_full_attn_query_chunk_tokens", type=int,
                    default=2048)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CorridorKey v2 runtime requires a CUDA GPU.")
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
    )
    model = load_model(paths, device="cuda")
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
