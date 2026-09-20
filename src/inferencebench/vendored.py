"""Locate the pinned upstream copy and the port's patches, and verify that the copy is untouched."""

import hashlib
import io
import json
import os
import tarfile
from pathlib import Path
from typing import Any

UPSTREAM = Path(__file__).parent / "upstream"
LOCK = Path(__file__).parent / "upstream.lock"
PATCHES = sorted((Path(__file__).parent / "patches").glob("*.patch"))


def upstream_lock() -> dict[str, Any]:
    """The recorded source, commit, and git tree hash of the vendored upstream copy."""
    return json.loads(LOCK.read_text())


def git_tree_hash(folder: Path) -> str:
    """Compute git's tree object hash for a directory, so the copy can be checked against the pinned commit offline."""

    def blob(path: Path) -> bytes:
        """Hash one file the way git stores it."""
        data = path.read_bytes()
        return hashlib.sha1(b"blob %d\0" % len(data) + data).digest()

    def tree(directory: Path) -> bytes:
        """Hash one directory; git sorts entries as if directory names ended in a slash."""
        entries = []
        for child in sorted(directory.iterdir(), key=lambda c: c.name + ("/" if c.is_dir() else "")):
            if child.is_dir():
                entries.append((b"40000", child.name.encode(), tree(child)))
            else:
                mode = b"100755" if os.access(child, os.X_OK) else b"100644"
                entries.append((mode, child.name.encode(), blob(child)))
        body = b"".join(mode + b" " + name + b"\0" + digest for mode, name, digest in entries)
        return hashlib.sha1(b"tree %d\0" % len(body) + body).digest()

    return tree(folder).hex()


def patch_digest() -> str:
    """Digest of every patch applied on top of upstream, part of the identity of any shared measurement."""
    digest = hashlib.sha256()
    for patch in PATCHES:
        digest.update(patch.name.encode() + b"\0" + patch.read_bytes() + b"\0")
    return digest.hexdigest()[:12]


def upstream_archive() -> bytes:
    """Pack the untouched upstream copy and the patches for installation inside a sandbox."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        archive.add(UPSTREAM, arcname="upstream")
        archive.add(LOCK, arcname="upstream.lock")
        for patch in PATCHES:
            archive.add(patch, arcname=f"patches/{patch.name}")
    return buffer.getvalue()


def scenario_directories() -> dict[str, str]:
    """Map each upstream scenario ID to its task directory by reading the vendored scenario files."""
    tasks = UPSTREAM / "src" / "eval" / "tasks"
    return {
        json.loads((folder / "scenario.json").read_text())["scenario_id"]: folder.name
        for folder in sorted(tasks.glob("inference_scenario_*"))
    }

