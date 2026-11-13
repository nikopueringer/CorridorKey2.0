"""Source-owned outputs, published as immutable generations.

Only the atomic colour manifest names completed outputs. Staging directories
and unreferenced generations are never discovered as render results.
"""
from __future__ import annotations

from contextlib import contextmanager
import glob
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

OWNER_FILE = ".corridorkey-source.json"
LOCK_FILE = ".corridorkey.lock"
OUTPUT_SUFFIXES = {"alpha": "alpha", "fg": "fg", "cutout": "cutout",
                   "checker": "checker_comp_despill*"}


class OutputSnapshot(dict):
    """Paths and their interpretation captured from the same committed manifest."""
    def __init__(self, paths: dict, metadata: dict, entries: dict | None = None):
        super().__init__(paths)
        self.metadata = metadata
        self.entries = entries if entries is not None else metadata.get("outputs", {})


def source_identity(source: str | os.PathLike) -> dict:
    canonical = os.path.normcase(str(Path(source).resolve()))
    return {"id": hashlib.sha256(os.fsencode(canonical)).hexdigest(), "path": canonical}


def read_manifest(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def contained_path(root: Path, relative: str) -> Path:
    """Resolve a manifest path without accepting traversal or escaping links."""
    if not isinstance(relative, str) or not relative:
        raise ValueError("Missing output path")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Output path must stay inside its output directory")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()) or resolved == root.resolve():
        raise ValueError("Output path escapes its output directory")
    return root / path


@contextmanager
def output_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / LOCK_FILE).open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            lock.seek(0)
            lock.write(b"\0")
            lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def atomic_json(path: Path, value: dict) -> None:
    fd, name = tempfile.mkstemp(prefix=path.name + ".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def owns_output_dir(root: Path, source: str | os.PathLike) -> bool:
    return read_manifest(root / OWNER_FILE).get("id") == source_identity(source)["id"]


def default_output_dir(source: str | os.PathLike, *, parent: Path, display_name: str) -> Path:
    legacy = Path(parent) / f"out_{display_name}"
    if owns_output_dir(legacy, source):
        return legacy
    return Path(parent) / f"out_{display_name}_{source_identity(source)['id'][:12]}"


def validate_destination(root: Path, source: str | os.PathLike) -> None:
    """Read-only preflight; ownership is checked again while claiming/committing."""
    root = Path(root)
    if not root.exists():
        return
    if not root.is_dir():
        raise ValueError(f"Output directory is not a folder: {root}")
    owner = read_manifest(root / OWNER_FILE)
    if owner.get("id") == source_identity(source)["id"]:
        return
    if owner or (root / OWNER_FILE).exists():
        raise ValueError(f"Output directory belongs to another source: {root}")
    if any(p.name != LOCK_FILE for p in root.iterdir()):
        raise ValueError(f"Output directory contains unowned results: {root}. "
                         "Choose a new output directory; existing results remain readable.")


def prepare_output_dir(root: Path, source: str | os.PathLike) -> None:
    """Claim an empty directory, or verify its existing source ownership."""
    root = Path(root)
    identity = source_identity(source)
    with output_lock(root):
        validate_destination(root, source)
        if not (root / OWNER_FILE).exists():
            atomic_json(root / OWNER_FILE, identity)


def merge_manifest(path: Path, manifest: dict) -> None:
    """Serialize read/merge/replace so concurrent subset renders keep entries."""
    with output_lock(path.parent):
        if "source" in manifest:
            owner = read_manifest(path.parent / OWNER_FILE)
            if owner.get("id") != manifest["source"].get("id"):
                raise ValueError("Output ownership changed before publication")
        old = read_manifest(path)
        merged = dict(manifest)
        outputs = old.get("outputs", {})
        outputs = dict(outputs) if isinstance(outputs, dict) else {}
        for key, entry in outputs.items():
            if isinstance(entry, dict) and "native_fps" not in entry and "native_fps" in old:
                outputs[key] = dict(entry, native_fps=old["native_fps"])
        outputs.update(manifest.get("outputs", {}))
        merged["outputs"] = outputs
        atomic_json(path, merged)


class OutputTransaction:
    def __init__(self, root: Path, source: str | os.PathLike, clip_name: str):
        self.root = Path(root).resolve()
        self.source = source_identity(source)
        try:
            stat = Path(source).stat()
            self.source_stat = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        except OSError:
            self.source_stat = None
        prepare_output_dir(self.root, source)
        self.manifest_path = self.root / f"{clip_name}_color.json"
        token = uuid.uuid4().hex
        self.stage = contained_path(self.root, f".staging/{token}")
        self.generation = contained_path(self.root, f"renders/{token}")
        self.stage.mkdir(parents=True)
        self.committed = False

    def commit(self, manifest: dict, expected_frames: int) -> dict:
        entries = manifest.get("outputs", {})
        if not entries:
            raise ValueError("No outputs were rendered")
        published = dict(manifest, schema_version=2, source=self.source)
        published["outputs"] = {}
        for key, entry in entries.items():
            path = contained_path(self.stage, entry["path"])
            if path.is_dir():
                count = len(list(path.glob("*.exr")))
                if count != expected_frames:
                    raise RuntimeError(f"Incomplete {key} output: {count}/{expected_frames} frames")
            elif not path.is_file() or not path.stat().st_size:
                raise RuntimeError(f"Missing or empty {key} output: {path}")
            published["outputs"][key] = dict(
                entry, path=(self.generation / path.relative_to(self.stage)).relative_to(self.root).as_posix(),
                frame_count=expected_frames, source_stat=self.source_stat)
        # A crash before manifest replacement leaves an unreferenced generation;
        # the previously committed manifest and its pixels stay untouched.
        self.generation.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.stage, self.generation)
        merge_manifest(self.manifest_path, published)
        self.committed = True
        return published

    def abort(self) -> None:
        # Never delete completed generations: another viewer may hold them open.
        if self.stage.exists():
            shutil.rmtree(self.stage)


def read_outputs(root: Path, source: str | os.PathLike, clip_name: str) -> dict[str, Path]:
    """Find committed outputs, retaining read support for unowned flat renders."""
    root = Path(root)
    if not root.is_dir():
        return {}
    owner = read_manifest(root / OWNER_FILE)
    if (root / OWNER_FILE).exists() and owner.get("id") != source_identity(source)["id"]:
        return {}
    metadata = read_manifest(root / f"{clip_name}_color.json")
    entries = metadata.get("outputs", {})
    entries = entries if isinstance(entries, dict) else {}
    found = {}
    if owner or metadata.get("schema_version") == 2:
        # Once a directory uses transactions, broken/missing metadata must not
        # expose uncommitted files through a fallback mtime/glob scan.
        saved_source = metadata.get("source")
        if (not owner.get("id") or metadata.get("schema_version") != 2
                or not isinstance(saved_source, dict) or saved_source.get("id") != owner["id"]):
            return {}
        for key, entry in entries.items():
            if key not in OUTPUT_SUFFIXES or not isinstance(entry, dict):
                continue
            try:
                path = contained_path(root, entry.get("path"))
                if Path(entry["path"]).parts[0] != "renders":
                    continue
                if path.exists():
                    found[key] = path
            except (ValueError, OSError, IndexError):
                continue
        return OutputSnapshot(found, metadata, {key: entries[key] for key in found})
    stem = glob.escape(clip_name)
    for key, suffix in OUTPUT_SUFFIXES.items():
        candidates = [p for p in root.glob(f"{stem}_{suffix}.*")
                      if p.suffix.lower() in (".mp4", ".mov")]
        candidates += [p for p in root.glob(f"{stem}_{suffix}_exr") if p.is_dir()]
        candidates = [p for p in candidates if p.resolve().is_relative_to(root.resolve())]
        if candidates:
            entry = entries.get(key, {})
            recorded = entry.get("path") if isinstance(entry, dict) else None
            completed = next((p for p in candidates if p.name == recorded), None)
            found[key] = completed or max(candidates, key=lambda p: p.stat().st_mtime)
    matched = {}
    for key, path in found.items():
        entry = entries.get(key)
        if isinstance(entry, dict) and entry.get("path", path.name) == path.name:
            matched[key] = entry
    return OutputSnapshot(found, metadata, matched)


def output_metadata(clip_name: str, outputs: dict[str, Path],
                    output_root: Path | None = None) -> tuple[dict, dict]:
    """Match profiles/FPS to actual paths, for legacy and generation outputs."""
    if isinstance(outputs, OutputSnapshot):
        return outputs.metadata, outputs.entries
    matched = {}
    global_metadata = {}
    for key, path in outputs.items():
        parent = path.parent
        roots = [output_root] if output_root is not None else [parent]
        if output_root is None and parent.parent.name == "renders":
            roots.insert(0, parent.parent.parent)
        for root in roots:
            metadata = read_manifest(root / f"{clip_name}_color.json")
            entries = metadata.get("outputs", {})
            entry = entries.get(key) if isinstance(entries, dict) else None
            if not isinstance(entry, dict):
                continue
            try:
                recorded = contained_path(root, entry.get("path", path.name))
                if recorded.resolve() != path.resolve():
                    continue
            except (ValueError, OSError):
                continue
            matched[key] = entry
            global_metadata = metadata
            break
    return global_metadata, matched
