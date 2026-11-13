# Apple Silicon community app and OpenFX

This experimental app combines David Coelho's Qt desktop UI (PR #8) with
Joseph Slinker and Parker Wightman's MPS support (PR #15). It is separate from
Corridor Digital's paid plugin. The desktop app also supplies the local MPS
worker for the separately packaged experimental macOS OpenFX bridge.

## Install and run

1. Use an Apple Silicon Mac running **macOS 15 or later**. The upstream MPS
   author tested a 24 GB M2; smaller-memory machines and this combined build
   have not been validated with a full model on a physical Mac.
2. Install [Homebrew](https://brew.sh), if needed, then run `brew install ffmpeg`.
   FFmpeg and FFprobe are external prerequisites; Python and Python dependencies
   are included in the app. Finder launches also search `/opt/homebrew/bin`.
3. Unzip `CorridorKey-macOS-arm64.zip`, move `CorridorKey.app` into Applications,
   and open it. The build is ad-hoc signed, **not Developer ID signed or
   notarized**. If macOS blocks opening it, follow Apple's per-app
   [Privacy & Security instructions](https://support.apple.com/en-us/102445)
   after checking the source/build provenance. Do not disable Gatekeeper.
4. Click **Download models** on first launch. About 5 GB is fetched from the
   original model publishers. Allow at least 12 GB free for downloads and
   temporary files. Models remain outside the app in
   `~/Library/Application Support/CorridorKey2.0`, so replacing the app does
   not redownload them. Interrupted downloads can be retried.
5. Add a clip, select outputs and click **Run queue**. The queue uses the existing CLI
   and automatically selects MPS. Start with a short clip and automatic memory
   settings. Results go next to the source unless another destination is set.

The first package uses standard PyTorch MPS attention. The optional native
MPSGraph extension is not bundled because it changes floating-point rounding.
Permanent BF16 weight storage from PR #9 remains CUDA-only; Metal retains
PR #15's original storage policy. CUDA/MPS pixel parity is not claimed.

Logs: `~/Library/Logs/CorridorKey2.0/launcher.log`. The desktop also shows
per-clip processing errors. Model processing is local after installation.

## Install OpenFX for Resolve

Use the app and OFX archives from the same CI build. The OFX archive is
`CorridorKeyMacOS-OFX-arm64.zip`; it does not include Python or model weights.

1. Install the app and models as above. Quit Resolve before installing the effect.
2. Extract the OFX archive and run `Install OFX.command`. It installs only the
   community `CorridorKeyMacOS.ofx.bundle` into `/Library/OFX/Plugins`; macOS
   requests an administrator password for that system folder. Existing bundles
   are refused rather than silently replaced. The commercial plugin is untouched.
3. Start CorridorKey and click **Start OFX engine**. Wait for the ready message.
4. Restart Resolve and add **CorridorKey 2.0 Community** from the **Keyer** group.
   Keep CorridorKey open while using the effect. Use the input-colour and alpha
   setup described in the [OpenFX guide](../../ofx/README.md).

The plugin is ad-hoc signed and not notarized, just like the app. It performs
single-frame keying; use the standalone queue for temporal processing and hints.
Maximum image size is 16,777,216 pixels and 8192 per side. An 8740×5760 source
must be processed at a smaller timeline resolution or through the standalone
queue using a supported input codec.

macOS CI exercises the compiled bundle in a mock OFX host, same-user socket
credentials and Python worker lifecycle/cancellation/recovery. An actual
macOS Resolve render and full-model MPS inference have not yet been verified.

## Reproduce the app

On an Apple Silicon Mac, install Xcode Command Line Tools and `uv`, commit the
source tree you want to package, then run:

```sh
./scripts/build_macos.sh
```

Outputs are `dist/CorridorKey.app` and `dist/CorridorKey-macOS-arm64.zip`.
Existing outputs are refused unless `--force` is explicitly passed. The script
exports **committed HEAD**, bundles relocatable CPython 3.12, installs PyTorch
2.9.1 / torchvision 0.24.1 plus the project requirements, builds an arm64
launcher and ad-hoc signs the bundle. It does not bundle weights or FFmpeg.

The `Community desktop builds` GitHub Actions workflow uses a macOS 15 arm64
runner. It extracts the zip to a different directory containing spaces,
removes the original build directory, verifies signatures, starts the native
launcher in self-test mode, checks CLI imports/help and runs CPU-compatible
regression tests. This proves packaging and startup, **not full-model MPS
inference or Resolve compatibility**. MPS-only tests may be skipped when the
hosted runner has no usable Metal GPU. The workflow uploads the app archive
only after those checks pass, and also compiles, tests, signs and packages the
arm64 OFX bundle.

The initial arm64 CI build passed its relocation/signature/UI checks and
41 tests, including available Metal tests; 5 optional native-attention tests
were skipped. Full-model video inference in the packaged app remains untested.

## Upstream credit and review

Original commits and licenses are retained. See [PR review](../../docs/PR_REVIEW.md)
and [third-party notices](../../THIRD_PARTY_NOTICES.md). Mac performance figures
in the main README describe the original PR author's runs; they are not a
benchmark of this combined app.
