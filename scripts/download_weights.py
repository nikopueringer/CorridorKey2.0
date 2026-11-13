#!/usr/bin/env python3
"""Download the CorridorKey v2 weights and the foundation models it depends on.

CorridorKey's own weights come from the CorridorKey Hugging Face repository.
The three frozen foundation models are fetched from their original publishers
at pinned revisions, under their own licenses (see THIRD_PARTY_NOTICES.md):

- C-RADIOv4-SO400M  (NVIDIA, NVIDIA Open Model License)
- MoGe-2 ViT-B      (Microsoft, MIT)
- RVM small         (Google DeepMind, CC-BY 4.0)

Files land in `weights/` and `vfms/` under the package root, which is where
the runtime looks for them. Re-running skips anything already present.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import shutil
import ssl
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

CORRIDORKEY_REPO = "nikopueringer/CorridorKey2.0"
CORRIDORKEY_FILES = ("corridorkey_v2.safetensors", "foundation_stats.safetensors")

CRADIO_REPO = "nvidia/C-RADIOv4-SO400M"
CRADIO_REVISION = "c0457f5dc26ca145f954cd4fc5bb6114e5705ad8"

MOGE_REPO = "Ruicheng/moge-2-vitb-normal"
MOGE_REVISION = "ca5f0e07ff01d3e5a364c1d954ed12ee1814b368"

RVM_FILE = "pretrain_rvm_small16_256_204031069.npz"
RVM_URL = f"https://storage.googleapis.com/representations4d/checkpoints/{RVM_FILE}"
RVM_MD5_B64 = "ZJb6kj8+tasiJbVeplXu7Q=="


def _md5_b64(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode()


def download_corridorkey(root: Path, repo: str, revision: str | None) -> None:
    for name in CORRIDORKEY_FILES:
        print(f"[corridorkey] {name}")
        hf_hub_download(repo, name, revision=revision, local_dir=root / "weights")


def download_cradio(root: Path) -> None:
    print(f"[c-radio] {CRADIO_REPO}@{CRADIO_REVISION[:7]}")
    snapshot_download(
        CRADIO_REPO, revision=CRADIO_REVISION,
        local_dir=root / "vfms/cradio/C-RADIOv4-SO400M",
        # The half-precision .pth.tar duplicates model.safetensors and is unused.
        ignore_patterns=["*.pth.tar"],
    )


def download_moge(root: Path) -> None:
    print(f"[moge] {MOGE_REPO}@{MOGE_REVISION[:7]}")
    hf_hub_download(MOGE_REPO, "model.pt", revision=MOGE_REVISION, local_dir=root / "vfms/moge")


def _ssl_context() -> ssl.SSLContext:
    """Return an SSL context, falling back to certifi if default CA paths are missing."""
    if ssl.get_default_verify_paths().cafile is None:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    return ssl.create_default_context()


def download_rvm(root: Path) -> None:
    dst = root / "vfms/rvm" / RVM_FILE
    if dst.is_file() and _md5_b64(dst) == RVM_MD5_B64:
        print(f"[rvm] {RVM_FILE} already present")
        return
    print(f"[rvm] {RVM_URL}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(RVM_URL, context=_ssl_context()) as response, open(tmp, "wb") as f:
        shutil.copyfileobj(response, f, length=1 << 20)
    if _md5_b64(tmp) != RVM_MD5_B64:
        tmp.unlink()
        raise RuntimeError(f"checksum mismatch downloading {RVM_URL}")
    tmp.replace(dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package_root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--corridorkey_repo", default=CORRIDORKEY_REPO)
    ap.add_argument("--corridorkey_revision", default=None)
    ap.add_argument("--skip_corridorkey", action="store_true",
                    help="Only fetch the third-party foundation models.")
    args = ap.parse_args()

    root = Path(args.package_root).resolve()
    if not args.skip_corridorkey:
        download_corridorkey(root, args.corridorkey_repo, args.corridorkey_revision)
    download_cradio(root)
    download_moge(root)
    download_rvm(root)
    print(f"All assets are in place under {root}")


if __name__ == "__main__":
    main()
