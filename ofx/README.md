# CorridorKey community Linux app and OpenFX bridge

An experimental Linux x86-64 build around the public CorridorKey 2.0 model.
The desktop UI continues David Coelho's [PR #8](https://github.com/nikopueringer/CorridorKey2.0/pull/8).
This community contribution is separate from the official paid plugin.

## Setup

Requirements: a supported NVIDIA GPU and working driver, ffmpeg, `uv`, and
around 15 GB free for the Python environment, CUDA libraries and model weights.
Rendering needs additional scratch/output space. Setup uses Python 3.12 and
the upstream-tested PyTorch 2.9.1 / CUDA 12.8.

```bash
./scripts/setup_linux.sh
./gui/CorridorKey.sh
# Optional applications-menu entry:
venv/bin/python -m gui.shortcut
```

If `uv` is missing, install it with `python3 -m pip install --user uv` and add
`~/.local/bin` to PATH. Setup installs into `venv/` and downloads models from
the original publishers. It does not install system packages. The GUI launcher
does not install packages automatically.

Drag clips or image-sequence folders into the queue. Choose hints, memory
settings and outputs in the inspector. The queue retains upstream temporal
Hann windows, automatic VRAM selection and first-frame/full-video hints.
EXR outputs are scene-linear; cutout MOV files are ProRes 4444 with straight
alpha. Import these rendered outputs into Resolve for temporal work.

## Build and install OpenFX

Requires a C++17 compiler, CMake, and OpenFX headers. The default header path
uses the SDK bundled with Resolve. Headers from
[AcademySoftwareFoundation/openfx](https://github.com/AcademySoftwareFoundation/openfx)
can be supplied with `-DOFX_INCLUDE_DIR=/path/to/openfx/include`.

```bash
./scripts/build_ofx.sh
sudo ./scripts/install_ofx.sh
```

The installer copies only `CorridorKeyLinux.ofx.bundle` to `/usr/OFX/Plugins`
and refuses to overwrite an existing bundle. No model, Python, CUDA or SDK
files are copied into Resolve. To uninstall, remove only that bundle. Pass
another directory to the installer if your host uses another plugin location.

Without sudo, Resolve 21.2 also discovers the bundle through `OFX_PLUGIN_PATH`:

```bash
./scripts/install_ofx.sh "$HOME/.local/lib/OFX/Plugins"
OFX_PLUGIN_PATH="$HOME/.local/lib/OFX/Plugins${OFX_PLUGIN_PATH:+:$OFX_PLUGIN_PATH}" /opt/resolve/bin/resolve
```

The environment must be present when Resolve starts. A normal Resolve launch
will not pick up this user-local path unless configured separately.
`./scripts/resolve_with_ofx.sh` sets this path and launches Resolve for you.

1. Restart Resolve after installation.
2. In the CorridorKey GUI, click **Start Resolve engine** and wait for
   **Resolve engine ready**. Keep the app open while using the effect.
3. Add **CorridorKey 2.0 Community**, in the **Keyer** group.
4. Choose the correct **Input colour**: sRGB transfer / Rec.709 primaries, or
   scene-linear / Rec.709 primaries. Convert camera log, ACEScg, DaVinci Wide
   Gamut or other working spaces upstream and transform back downstream as
   needed. No colour metadata or primary-space conversion is applied.
5. Select keyed RGBA, matte or foreground view and adjust despill. Keyed RGBA
   is premultiplied. On the Color page, connect the node's alpha output to an
   Alpha Output when using the matte to composite the clip.
6. After starting/restarting an engine, press **Refresh / retry engine** if a
   previous render failed.

For headless use, `./scripts/resolve_engine.sh` starts the same worker. It
listens on `/tmp/corridorkey-<uid>/engine.sock`: private 0700 directory, 0600
socket, same-user credential checks, no TCP listener. Frames use isolated
scratch directories and float32 RGBA transport. One model request runs at a
time; multiple plugin instances share it. Starting the GUI queue stops the
engine owned by that UI to release GPU memory.

## Current limits

- OFX is a **single-frame** keyer. Temporal windows and hints remain available
  in the standalone queue, not inside this first OFX implementation.
- Inference is not real-time playback. Resolve's render cache can help with
  playback; each effect also keeps one exact input/result in memory.
- Linux x86-64, progressive frames and CPU-side float RGBA host buffers only.
  No CUDA zero-copy. Maximum 16,777,216 pixels and 8192 on either side; native
  UHD/DCI 4K fits, full 8K exceeds the protocol limit.
- Cancellation discards queued obsolete frames and checks between model
  phases/windows. An already executing CUDA operation cannot be preempted.
- Source alpha is multiplied with predicted alpha. Matte and foreground
  preview modes output opaque images.
- Failures are reported to the host rather than silently returning an unkeyed
  source. No commands or file paths are accepted through the frame protocol.
- Resolve Studio 21.2 on Linux was verified to discover/register the effect,
  create a Fusion instance and export matte and keyed RGBA through its timeline
  Deliver path. A synthetic 640×360 clip produced finite EXRs, grayscale matte
  output and keyed alpha spanning 0–1; the keyed result was visually inspected.
  This is a host smoke test, not a real-footage or colour-management validation
  matrix. Direct rendering of detached Fusion Saver graphs did not work in the
  headless harness; timeline rendering exercised the plugin successfully.

## Tests

```bash
uv pip install --python venv/bin/python pytest
QT_QPA_PLATFORM=offscreen venv/bin/python -m pytest tests -q
./scripts/build_ofx.sh
# With the CUDA engine running:
venv/bin/python scripts/smoke_test_ofx.py input.png /tmp/keyed.npy
```

Coverage includes OFX lifecycle, float buffer strides/orientation,
premultiplication, cache invalidation, unavailable/busy engines, malformed
replies, cancellation, private socket ownership, invalid requests, worker
recovery/locking, GUI process-group termination and scratch isolation, natural
frame order, and LUT preprocessing against the original float path.

## Community contributions

Original author commits are retained for:

- [#8 — David Coelho](https://github.com/nikopueringer/CorridorKey2.0/pull/8): desktop GUI.
- [#4 — Davidhanson90](https://github.com/nikopueringer/CorridorKey2.0/pull/4): natural sorting and uppercase extensions; GUI preview order is aligned here.
- [#9 — Reinbowl](https://github.com/nikopueringer/CorridorKey2.0/pull/9): BF16 Linear/Conv weight storage and frozen parameters.
- [#20 — MrFruitDude](https://github.com/nikopueringer/CorridorKey2.0/pull/20): 8-bit preprocessing LUTs. This branch adds a guard preserving the original frame-directory path for adjusted metadata.

Reviewed and deferred: #11 (OOM retry/profile policy), #18 (broad optimized
runtime), and #21 (consolidation explicitly not intended to merge). They need
a broader device/footage regression set than this integration.

CorridorKey remains credited to Corridor Digital and Zachary Harris. Existing
[licence terms](../LICENSE) and [third-party notices](../THIRD_PARTY_NOTICES.md)
remain in effect. Weights and third-party SDK files are not included in the
source contribution or OFX bundle.
