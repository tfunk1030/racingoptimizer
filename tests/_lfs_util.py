"""Helpers shared by tests that touch the git-lfs-tracked `ibtfiles/` corpus.

The IBT corpus is large (5 GB+ across ~120 sessions) and tracked in git-lfs.
On checkouts where `git lfs pull` hasn't run, the path-of-record is a
~130-byte pointer text file. Feeding that pointer through irsdk's binary
parser causes runaway memory allocation (the pointer text gets misread as
a multi-GB `session_info_len`). Tests that load real IBTs must skip
cleanly when the pointer is unmaterialised.
"""
from __future__ import annotations

from pathlib import Path

_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def is_unmaterialised_lfs_pointer(path: Path) -> bool:
    try:
        if path.stat().st_size > 4096:
            return False
        with path.open("rb") as fh:
            return fh.read(len(_LFS_POINTER_PREFIX)) == _LFS_POINTER_PREFIX
    except OSError:
        return False


def lfs_skip_message(path: Path) -> str:
    return (
        f"{path.name} is an unmaterialised git-lfs pointer; "
        "run `git lfs pull` before invoking IBT-loading tests"
    )
