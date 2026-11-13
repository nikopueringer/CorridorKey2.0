# CorridorKey 2.0

## Full macOS support on Apple Silicon

**Run CorridorKey 2.0 on your Mac with Metal GPU acceleration.**
This fork preserves the existing command-line tools and original model weights,
with memory controls for native 4K processing and optional native MPSGraph
attention. A complete native 4K clip has been validated on a 24 GB M2 Mac.

**[Get started on macOS →](#apple-silicon-mac)**

---

Welcome to CorridorKey 2.0! You may rest easy, weary artist. The green beast has been slain. The CorridorKey 2.0 model was designed and trained by Zachary Harris, in direct collaboration with Niko and Corridor Digital.

# Check out our official plugin!

The most requested feature we've heard this year- "Can you make a plugin for my favorite software?

...yes! Finally!

Led by Ale Alvaro, we now have a plugin (and stand-alone app) available for almost every media software package. A one-time purchase gets you access to ALL OF THEM.
**[corridorkey.com](https://corridorkey.com)**

...And for those of you that want to experiment, tinker, and build your own python tools, welcome to the CorridorKey 2.0 repo! You may use this model and software freely, for commercial or personal work. See the license for more details, but that's it in a nutshell. Enjoy!

Any business or research inquiries for Zac can be directed to z.brooks.harris@gmail.com


## Setup

### Apple Silicon Mac

Use native arm64 Python 3.12, a macOS version with Metal bfloat16 support
(macOS 14 or later), and `ffmpeg` on your `PATH`. From the repository root:

```sh
uv sync
uv run scripts/download_weights.py
uv run corridorkey --input /path/to/clip.mp4 --output_dir /path/to/output
```

The existing CLI automatically selects Metal when CUDA is unavailable.
Metal uses synchronous uploads and the CPU area-resize operator for dimensions
unsupported by Metal's adaptive pooling. The CUDA execution path is unchanged.
The Mac dependencies select a tested OpenCV decoder to avoid corrupted final
rows in YUV444 video with the older Apple Silicon wheel.

On macOS 15 or later, optional native MPSGraph attention reduces attention
memory and improves throughput. With Apple's command-line developer tools
installed, build it in the same Python environment before running inference:

```sh
uv run scripts/build_mps_attention.py
```

Rebuild after changing PyTorch or moving the environment. A missing or
incompatible library uses the bounded PyTorch path. Set
`CORRIDORKEY_MPS_NATIVE_ATTENTION=0` to disable native attention for comparisons.

Automatic memory selection reserves headroom for unified memory and reduces
foundation, attention, and MLP batches on smaller Macs. Existing explicit Hann
and low-VRAM settings retain control. Metal allocation limits remain enabled;
explicit `PYTORCH_MPS_HIGH_WATERMARK_RATIO` / `PYTORCH_MPS_LOW_WATERMARK_RATIO`
settings are respected.

A 302-frame 3840×2160 clip completed on a 24 GiB M2 with native attention:
3 h 58 min end to end, 11.96 GiB sampled inference-process peak (excluding
separate FFmpeg processes). Both output videos decoded all 302 frames at native
resolution. Model weights and resolution were preserved. This remains a slow
offline workflow; automatic settings can use shorter temporal context.
Native attention can introduce floating-point differences, and identical
CUDA/Metal pixels are not guaranteed.

### NVIDIA CUDA

You need an NVIDIA GPU (currently RTX 30-series or newer) with at least 4 GB of VRAM
for footage up to 1440p, or 6 GB for 4K, and `ffmpeg` on your PATH. On smaller cards,
see [Long Clips And Low VRAM](#long-clips-and-low-vram).

From the repository root, install dependencies and download the model weights
(about 5 GB). `uv` selects the CUDA 12.8 PyTorch wheel automatically on Linux and Windows:

```
uv sync
uv run scripts/download_weights.py
```

The download script fetches:

- the CorridorKey 2.0 weights from
  [Hugging Face](https://huggingface.co/nikopueringer/CorridorKey2.0) into
  `weights/`
- the three frozen foundation models CorridorKey builds on, from their original
  publishers, into `vfms/`: C-RADIOv4-SO400M (NVIDIA), MoGe-2 ViT-B
  (Microsoft), and RVM small (Google DeepMind)

After that everything loads from those local folders.

## Drag-And-Drop Launcher

The easiest way to run CorridorKey is to drag a clip, a frame-sequence folder,
or a folder of clips onto the launcher in the repository root:

- **Windows:** `CorridorKey_DRAG_CLIPS_HERE.bat`
- **Linux:** `CorridorKey_DRAG_CLIPS_HERE.sh` (or run it in a terminal and drop
  the clip into the terminal when asked)

It walks you through the inference mode, outputs, and GPU memory profile, then
runs the job. The launcher uses the `venv` created during setup if it exists.
It also picks up companion hint files next to the input automatically
(`<clip>FrameHint.png` for a first-frame hint, `<clip>_AlphaHint.mp4` for a
full hint video).

You can also start the same guided session directly:

```
uv run corridorkey-interactive /path/to/clip.mp4
```

## Quick Start

From the repository root:

```
uv run corridorkey \
  --input /path/to/input.mp4 \
  --output_dir /path/to/output
```

This runs native-resolution inference with no external hint. By default it
writes:

- `alpha`: predicted matte as a video preview.
- `fg`: predicted foreground RGB as a video preview.

Native resolution is always preserved. The model reflect-pads internally to its
required shape quantum and crops back to the original size for output.

## Choosing An Inference Mode

CorridorKey can run with no hints, a single first-frame hint, or a full hint
video. The right mode depends on what information you have and how much you want
the model to decide for itself.

### No Hint

```
uv run corridorkey \
  --input shot.mp4 \
  --output_dir out_no_hint
```

Use this when no matte or object hint is available. The model decides what the
foreground subject is.

Tradeoffs:

- Fastest setup because no hint asset is needed.
- Useful for quick tests and simple shots.
- Less controlled when multiple plausible subjects are present.
- The model may make a different subject-selection decision than you intended.

### First-Frame Hint

```
uv run corridorkey \
  --input shot.mp4 \
  --output_dir out_frame1_hint \
  --hint_first_frame frame_0001_matte.png \
  --hint_quality 0.95
```

Use this when you can provide a good matte for the first frame. The hint should
be a single-channel matte or an image with an alpha channel: white means subject,
black means background.

By default, Hann carry is enabled. This means the model uses its own predicted
auxiliary matte from overlapping Hann windows as a continuing hint signal after
the first explicit hint.

Tradeoffs:

- Usually the best practical mode when only a starting matte is available.
- Gives the model a clear object identity at the beginning of the shot.
- Can track through time without requiring a full hint video.
- Can drift if the shot is long, crowded, or has large occlusions.
- Larger Hann windows can improve continuity but cost more VRAM.

### Full Hint Video

```
uv run corridorkey \
  --input shot.mp4 \
  --output_dir out_full_hint \
  --hint_video hint_matte_video.mp4 \
  --hint_quality 0.95
```

Use this when you have a matte, roto pass, garbage matte, or other subject hint
for every frame.

Tradeoffs:

- Most controlled mode.
- Best when an external tool or artist pass already knows the intended subject.
- The output will tend to follow systematic bias in the hint, so a hint that is
  too eroded, too dilated, or attached to the wrong subject can steer the result
  in that direction.
- If you want the full hint video to be used directly on every pass, add
  `--no_carry_hint`. Otherwise, carried model hints are used on overlapping
  frames while explicit hints are still used for non-carried frames.

### Full Hint Video Without Carry

```
uv run corridorkey \
  --input shot.mp4 \
  --output_dir out_full_hint_no_carry \
  --hint_video hint_matte_video.mp4 \
  --hint_quality 0.95 \
  --no_carry_hint
```

Use this when the hint video is authoritative and you do not want the model's
previous-window predictions to feed back as hints.

Tradeoffs:

- More literal use of the provided hint video.
- Avoids feedback from an imperfect carried prediction.
- Gives up one of the temporal stabilization mechanisms used by the default
  setup.

## Hint Quality

`--hint_quality` tells the model how much to trust an explicit hint. It does not
change the hint image itself; it changes the conditioning value supplied to the
model.

Practical values:

- `0.95`: high-quality matte or artist-provided hint.
- `0.75`: rough but useful automated hint.
- `0.50` or below: loose hint, weak scribble-like mask, or unreliable source.

For carried Hann hints, use `--carry_hint_quality`. The default is `0.95`.

```
uv run corridorkey \
  --input shot.mp4 \
  --output_dir out_frame1_hint \
  --hint_first_frame frame_0001_matte.png \
  --hint_quality 0.95 \
  --carry_hint_quality 0.95
```

If a Hann window has no hints at all, the model receives the no-hint signal and
is asked to decide the subject. If a window has at least one hinted frame,
unhinted frames in that same window receive quality zero, meaning "there is a
hinted subject in this window, but this frame does not have its own explicit
hint."

## Hann Windows

Stage 1 runs on overlapping temporal windows and combines its hidden states with
a Hann-style taper. Stage 2 then streams the final full-resolution predictions
from those combined hidden states.

By default the window size is chosen automatically for each clip from the
free memory on your GPU and the clip's resolution, using the largest window
that fits (from `80/40` down to a single frame). The chosen values are printed at the start of the
run and saved in the settings log. To set them yourself:

```
--hann_chunk 80 --hann_stride 40
```

Passing `--hann_chunk` or `--low_vram` turns automatic selection off.

What the settings mean:

- `hann_chunk`: number of frames processed together by stage 1.
- `hann_stride`: distance between consecutive stage-1 windows.
- `80/40`: 50 percent overlap, the high-context maximum for a high-memory
  workstation GPU.
- `8/4`: lower VRAM and shorter temporal context, sometimes useful for memory
  pressure or quick testing.
- `1/1`: no stage-1 temporal window at all; each frame is keyed on its own.
  The lowest-memory setting.
- `48/24` or `24/12`: smaller alternatives if `80/40` is too heavy on a
  particular machine.
- Larger windows: often better temporal continuity, higher peak VRAM, more
  latency.
- Smaller windows: lower peak VRAM, weaker long-range temporal consistency.
- `96/48` was not stable at native 4K on the tested 96 GB card; `80/40`
  completed but used most of the available VRAM.

Example:

```
uv run corridorkey \
  --input shot.mp4 \
  --output_dir out_window_8 \
  --hann_chunk 8 \
  --hann_stride 4
```

## Outputs

Default outputs are `alpha` and `fg`.

Available output choices:

- `alpha`: matte preview.
- `fg`: raw foreground RGB prediction preview.
- `checker`: predicted foreground composited over a checkerboard with customizable despill applied.
- `cutout`: predicted foreground multiplied by predicted alpha (un-premultiplied straight color, despilled).

`raw_fg` and `foreground` are accepted aliases for `fg`.

Select outputs explicitly:

```
uv run corridorkey \
  --input shot.mp4 \
  --output_dir out_review \
  --outputs alpha fg checker cutout
```

Performance note: each additional output requires more GPU rendering work and
more ffmpeg encoding. For fastest normal use, keep the default `alpha fg`. Add
`checker` or `cutout` for review renders.

### Despill Strength

Use `--despill_strength` (value from `0.0` for no despill to `1.0` for full despill, defaults to `0.5`) to configure the despill amount applied to the `checker` and `cutout` outputs.

```
uv run corridorkey \
  --input shot.mp4 \
  --output_dir out_despill \
  --outputs checker cutout \
  --despill_strength 0.75
```

### Cutout Linear

By default, the straight color for the `cutout` output is in sRGB color space. If you want to export the cutout as straight scene-linear RGB instead of sRGB (common for linear VFX compositing pipelines), pass:

```
--cutout_linear
```

`--stage2_batch` controls how many final-resolution frames are run and rendered
together in stage 2. The default is `1`, which uses the least memory and was
no slower than larger batches in our tests.

- Higher values use more VRAM in stage 2, most noticeably at 4K.
- At 4K, stage 1 is usually the practical peak-memory path, so Hann window size
  often matters more than `--stage2_batch`.

## HDR EXR Output

By default, all outputs are ffmpeg-encoded preview videos. That path is
display-referred: values are clipped to `[0, 1]` and quantized for video
encoding (10-bit `yuv444p10le` x264 for `alpha`/`fg`/`checker`, ProRes 4444
for `cutout`). HDR values above 1.0 do not survive the video path.

For full HDR float output, pass:

```
--output_format exr
```

Each selected output is then written as an OpenEXR image sequence in its own
subdirectory of the output directory (e.g. `<clip>_fg_exr/fg_000000.exr`),
instead of a video file:

- `fg`, `cutout`, and `checker` color is unclamped scene-linear RGB. Values
  above 1.0 and negative values predicted by the model are preserved.
- `alpha` is a single-channel matte, clamped to `[0, 1]` (a matte is
  display-independent coverage, not HDR data).
- `cutout` is straight (un-premultiplied) linear RGB plus alpha in one RGBA
  EXR. `--cutout_linear` is irrelevant here: EXR output is always scene-linear.
- `--despill_strength` applies to `cutout` and `checker` as usual, computed in
  linear light.

EXR pixel type defaults to `half` (float16, the VFX interchange standard).
Half cannot represent values above 65504, so half output is clamped to the
finite float16 range to avoid `inf` pixels. For extreme HDR values or full
float32 files, pass:

```
--exr_dtype float
```

Effective precision note: the network itself runs in bfloat16 (in the asinh
HDR domain), which bounds output accuracy to roughly 1–2% relative error
through most of the HDR range regardless of the EXR pixel type. `half` adds
at most ~0.05% on top of that; `float` adds none.

Typical HDR round trip from linear EXR plates:

```
uv run corridorkey \
  --input /path/to/linear_exr_frames \
  --frame_dir_linear \
  --frame_dir_fps 24 \
  --output_format exr \
  --outputs alpha fg cutout \
  --output_dir out_hdr
```

Note that `--frame_dir_linear` matters on the input side: without it, EXR
frames are treated as display-coded and clipped to `[0, 1]` before the model
sees them. The runtime prints a warning if EXR input is read without the flag.
The ffmpeg encoding options below only apply to `--output_format video`.

## Video Encoding

Outputs are written through ffmpeg. Defaults:

```
--ffmpeg_codec libx264 --crf 12 --preset medium --pix_fmt yuv444p10le
```

Bit depth in the video path: frames are rendered in fp32 and piped to ffmpeg
as 16-bit raw video. The x264 preview outputs default to 10-bit 4:4:4
(`yuv444p10le`); pass `--pix_fmt yuv444p` if a playback target needs 8-bit
H.264. The `cutout` output is always ProRes 4444 with alpha, encoded from
10-bit input (decodes as 12-bit `yuva444p12le`). Video containers remain
display-referred and clipped to [0, 1] — use `--output_format exr` for
unclipped HDR.

Override these as needed for your pipeline. For example:

```
uv run corridorkey \
  --input shot.mp4 \
  --output_dir out_encoded \
  --ffmpeg_codec libx264 \
  --crf 10 \
  --preset slow \
  --pix_fmt yuv444p
```

## Long Clips And Low VRAM

You normally do not need to set anything here: with no memory flags,
CorridorKey measures the free memory on your GPU and picks the Hann window and
low-VRAM mode for each clip. As a guide, this is roughly what it selects
(window as chunk/stride):

| GPU memory | 720p | 1080p | 1440p | 4K |
|---|---|---|---|---|
| 4 GB | 4/2 low-VRAM | 2/1 low-VRAM | 1/1 low-VRAM | not enough memory |
| 6 GB | 24/12 low-VRAM | 8/4 low-VRAM | 4/2 low-VRAM | 2/1 low-VRAM |
| 8 GB | 24/12 low-VRAM | 8/4 low-VRAM | 8/4 low-VRAM | 4/2 low-VRAM |
| 12 GB | 80/40 low-VRAM | 24/12 low-VRAM | 8/4 | 8/4 low-VRAM |
| 16 GB | 80/40 | 48/24 low-VRAM | 24/12 low-VRAM | 8/4 low-VRAM |
| 24 GB | 80/40 | 48/24 | 48/24 low-VRAM | 8/4 |
| 32 GB | 80/40 | 80/40 | 48/24 | 24/12 |
| 48 GB | 80/40 | 80/40 | 80/40 | 48/24 low-VRAM |

On small cards it goes down to windows of only a few frames, or a single
frame, which gives up temporal context in stage 1 to save memory. Results
remain good, with somewhat less frame-to-frame consistency than larger windows.

These come from measurements on synthetic test clips; other software using the
GPU reduces what is available. If you hit a CUDA out-of-memory error, pass
`--low_vram` with a smaller `--hann_chunk` yourself.

### Optimized mode

The launcher's default memory profile, and `--optimized` on the command line,
run the same model in less memory and less time. With the same window size the
output is identical; in auto mode the saved memory goes into longer windows:

| GPU memory (free) | 720p | 1080p | 1440p | 3K | 4K |
|---|---|---|---|---|---|
| 6 GB laptop (4.95 GB) | 48/24 | 40/20 | 28/14 | 16/8 | 10/5 |
| 8 GB (6.95 GB) | 64/32 | 56/28 | 40/20 | 24/12 | 16/8 |
| 12 GB (10.95 GB) | 80/40 normal | 80/40 | 64/32 | 40/20 | 28/14 |
| 16 GB (14.95 GB) | 80/40 normal | 80/40 normal | 80/40 | 64/32 | 40/20 |
| 24 GB (22.95 GB) | 80/40 normal | 80/40 normal | 80/40 normal | 80/40 | 64/32 |

Windows picks, low-VRAM mode unless marked normal; the shipped settings on a
6 GB card are 8/4 at 720p and 1080p, 4/2 at 1440p and 2/1 at 3K and 4K.
Linux gets a step or two more from 1440p up (a 6 GB card runs 32/16 at 1440p,
24/12 at 3K and 16/8 at 4K): PyTorch's allocator has expandable segments there
and not on Windows.

Choose "Auto, original" in the launcher, or leave out `--optimized`, for the
original settings. The measurements, and how the output was checked, are in
https://github.com/nikopueringer/CorridorKey2.0/pull/18.

`--optimized` also keeps the GPU busy while the host reads and writes
frames, and starts faster. Each of these leaves every output file
byte-identical and can be used on its own:

- `--pinned_io`: frames are quantized to 16 bits on the GPU (the same values
  as before for every possible input) and copied through pinned memory one
  frame behind, so the GPU starts the next frame instead of waiting.
- `--lean_reads`: in low-VRAM mode, stages that only feed the model skip
  preparing each frame's display view (for linear EXR input that is an sRGB
  conversion of every pixel).
- `--read_ahead`: stage 1's first window is read while the prepass runs;
  later windows and stage 2's frames are read on worker threads.
- `--fast_load`: the model is built without first filling its weights with
  random values, since every one is then loaded from the checkpoints.
- `--decode_once`: each frame is decoded and converted once; the later
  stages read the model's input back from a temporary file (about 13 MB per
  1080p frame and 53 MB per 4K frame in `--temp_dir`, twice that in normal
  mode; skipped when the temp folder is short of space). Most useful at 4K
  and with EXR input.
- `--gpu_prep`: with `--decode_once` and a frame folder, each frame's
  color conversions run on the GPU. For 8-bit images and half-float EXRs
  the GPU gives the CPU's bits for every possible value; EXRs stored as
  32-bit float, where a few values round differently, stay on the CPU.
- `--async_baton`: in low-VRAM mode, stage 1's windows run back to back on
  the GPU; each window's output is copied back and blended into the
  finished frames while the GPU runs the next one.
- `--exr_workers N`: with `--output_format exr`, N frames per output are
  compressed and written at once (default 1; `--optimized` uses a quarter of
  the CPU cores, 2 to 8). Each file is still written whole by one thread from
  the same pixels, so the files are the same. ZIP-compressing a 4K frame
  takes a CPU core a few seconds, which otherwise sets the pace at 4K.

These were checked on Volta, Turing, Ampere, Ada, Hopper and Blackwell GPUs:
at the same window, `--optimized` writes the original's files on each. On
some cards the original's own output varies slightly from run to run
(cuDNN's default algorithms on an RTX 4090, the carry hint on an RTX 5080);
with deterministic cuDNN on the 4090 and the carry hint off on the 5080,
both repeat and agree.

`--fast` is `--optimized` with more threads per video encoder (or pass
`--ffmpeg_threads N` yourself). The keyed frames are identical, but x264's
output depends on its thread count, so the encoded files differ slightly
from the default two-thread encode. On fast GPUs at HD the encoder often
sets the pace and `--fast` is 5-10% faster; at 4K on a CPU with 16 cores or
fewer the extra encoder threads compete with frame preparation and can make
it slower, so measure on your machine. On a 6 GB RTX 3060 Laptop, where the
GPU sets the pace, it measured 1-3% faster at 1080p and 1440p.

The normal path streams video and does not intentionally hold an entire long
clip in RAM. To force low-VRAM mode yourself, add:

```
uv run corridorkey \
  --input long_shot.mp4 \
  --output_dir out_low_vram \
  --low_vram \
  --temp_dir /path/with/free/space
```

Low-VRAM mode is exact staged inference:

- RVM still runs chronologically through the clip with continuous recurrent
  state.
- C-RADIO and MoGe features are precomputed to temporary memmaps.
- Stage 1 runs Hann windows.
- Stage 2 streams final frames.

Tradeoffs:

- Lower peak VRAM.
- More disk I/O and temporary storage.
- Usually slower.
- Use a temp directory with enough free space for feature memmaps.

Advanced memory knobs:

- `--low_vram_foundation_chunk`: number of frames per VFM precompute chunk.
- `--low_vram_enc_frame_chunk`: stage-1 encoder frame chunking.
- `--low_vram_mlp_chunk_tokens`: MLP token chunking.
- `--low_vram_swin_window_batch`: Swin/window processing chunking.
- `--low_vram_full_attn_query_chunk_tokens`: query chunking for full attention.

Use the defaults unless you are actively tuning memory behavior.

## Frame Directories

`--input` can be a video file or a directory of frames. Frame directories are
read in sorted filename order. Supported extensions are EXR, PNG, JPG, and JPEG.

```
uv run corridorkey \
  --input /path/to/frames \
  --frame_dir_fps 24 \
  --output_dir out_frames
```

Frame directories are treated as display-coded by default. If the incoming image sequence is scene-linear RGB (e.g. for linear EXR sequences), pass:

```
--frame_dir_linear
```

Example:

```
uv run corridorkey \
  --input /path/to/linear_exr_frames \
  --frame_dir_fps 24 \
  --frame_dir_linear \
  --output_dir out_exr
```

## Partial Clips

For quick tests, process only part of a clip:

```
uv run corridorkey \
  --input shot.mp4 \
  --output_dir out_test \
  --num_frames 100 \
  --start_mode middle
```

`--start_mode` choices:

- `begin`: start from the first frame.
- `middle`: take a centered segment.
- `random_middle`: choose a deterministic random middle segment using `--seed`.

## Input Color Handling

The runtime does not inspect or convert color metadata. It uses a simple,
explicit input contract:

- Normal video files are decoded with OpenCV/ffmpeg into RGB values in `[0, 1]`
  and treated as display-coded sRGB/Rec.709-like footage.
- PNG/JPG frame directories are treated the same way: display-coded RGB in
  `[0, 1]`.
- Frame directories are also treated as display-coded by default.
- If frames are already scene-linear RGB, pass `--frame_dir_linear`.

For display-coded inputs, the runtime converts RGB to an internal scene-linear
working value with the package gamma function, then applies the model's signed
asinh HDR encoding. Model outputs are decoded back through the inverse path for
video previews.

For `--frame_dir_linear`, the incoming RGB values are passed as scene-linear
values directly into the model's signed asinh encoding. This is the path to use
for unclipped linear image sequence workflows (e.g. EXR sequences).

The foundation models are different: C-RADIO, MoGe, and RVM receive SDR-style
RGB proxies because those models expect normal image-like inputs. They do not
receive unclipped HDR values.

What this runtime does not currently do:

- It does not automatically convert ACEScg, Rec.2020, LogC, S-Log, V-Log, PQ,
  HLG, or camera-raw color spaces.
- It does not use video color metadata to choose a transfer function or primary
  matrix.
- It does not preserve HDR values from ordinary encoded video containers unless
  those values have already been converted into a supported frame input path.

Practical guidance:

- For ordinary sRGB/Rec.709-looking MP4/ProRes review footage, use the video
  directly.
- For linear EXR plates, use a frame directory and pass
  `--frame_dir_linear`.
- For Log, PQ/HLG, ACEScg, Rec.2020, or other managed-color sources, convert
  upstream into either display-coded RGB for the video path or scene-linear RGB
  frames for the linear path.

## Brightness And Contrast Controls

Leave these controls at default for standard inference:

```
--linear_brightness 1.0
--linear_contrast 1.0
--linear_contrast_pivot 0.18
```

The brightness and contrast options apply an experimental linear-light adjustment
after display-coded inputs have been converted to linear light and before the
model's asinh encoding. The inverse adjustment is applied to the predicted
foreground before writing preview outputs. These controls are useful for
controlled tests, not for ordinary production use.

## Failure Behavior

The runtime intentionally does not silently switch modes. Missing weights,
missing foundation assets, missing ffmpeg, unsupported inputs, and load
mismatches raise errors instead of falling back to a different behavior.

## Common Recipes

No hint, fastest standard outputs:

```
uv run corridorkey --input shot.mp4 --output_dir out_no_hint
```

First-frame hint with default carry:

```
uv run corridorkey \
  --input shot.mp4 \
  --output_dir out_frame1 \
  --hint_first_frame matte_frame_0001.png \
  --hint_quality 0.95
```

Full hint video, trusting the provided hint on every pass:

```
uv run corridorkey \
  --input shot.mp4 \
  --output_dir out_full_hint \
  --hint_video hint.mp4 \
  --hint_quality 0.95 \
  --no_carry_hint
```

Review render with checker and cutout:

```
uv run corridorkey \
  --input shot.mp4 \
  --output_dir out_review \
  --hint_first_frame matte_frame_0001.png \
  --outputs alpha fg checker cutout
```

Lower-memory 8-frame windows:

```
uv run corridorkey \
  --input shot.mp4 \
  --output_dir out_lowmem_window \
  --hint_first_frame matte_frame_0001.png \
  --hann_chunk 8 \
  --hann_stride 4
```

## Licensing and Permissions

Use CorridorKey for whatever you'd like, including processing footage for
commercial projects. What you make with it is yours.

- You may not sell this tool, or charge for access to its code or model, and
  you may not offer it as a paid API or inference service, unless given permission by Corridor Digital.
- You may build on the code and model, but what you release must be freely
  accessible, stay under this same license, and keep the CorridorKey name.
  Donationware and pay-what-you-want are fine.
- Organizations with more than US $20 million in annual revenue need to
  purchase a seat of the CorridorKey plugin for each person using this tool.
  A seat covers both the plugin and this model and software. See
  [corridorkey.com](https://corridorkey.com).
- If you run a commercial software package and want to integrate CorridorKey,
  email contact@corridordigital.com. We're easy to work with.

The full terms are in [LICENSE](LICENSE). The license is a variation of
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) with
additional terms.

## Contributing

Install the project and its dev dependencies, then run the test suite before and after your changes:

```sh
uv sync --extra dev
uv run python -m pytest tests/ -v
```

The tests cover auto-VRAM selection, frame path ordering, MPS attention, RMSNorm dtype preservation, and window batching. They run on CPU, MPS, or CUDA depending on what is available. CUDA-only tests are skipped automatically on macOS. All tests should pass (some skip on hardware that is not present) before a PR is submitted.

## Acknowledgements

CorridorKey builds on three foundation models. Please go show their creators
some support:

- [C-RADIO](https://github.com/NVlabs/RADIO) by NVIDIA, under the NVIDIA Open
  Model License
- [MoGe](https://github.com/microsoft/MoGe) by Microsoft, under the MIT license
- [RVM](https://github.com/google-deepmind/representations4d) by Google
  DeepMind, code under Apache 2.0 and weights under CC-BY 4.0

Details are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
