"""Flag the manifest's held-out sessions as gate-only in the catalog.

A fresh ``optimize learn`` ingests every IBT with ``held_out=0`` (the
catalog's ``upsert_session`` only *preserves* an existing flag; Day-0 prep
set it manually). Any environment that rebuilds the corpus from scratch —
the weekly CI job, a new workstation — must therefore re-apply the flags
before running the accuracy gates, or the per-car fits silently train on
the held-out IBTs (a leak `verify_holdout.sh` then reports as exit 2).

Session IDs are parsed from ``docs/physics-rebuild/holdout.sha256``'s
``session_id=`` comments so the manifest stays the single source of truth.

Run: ``uv run python scripts/mark_holdout_sessions.py``
Exit codes: 0 flags applied (or already set); 1 manifest/catalog missing.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "docs" / "physics-rebuild" / "holdout.sha256"

_SESSION_ID_RE = re.compile(r"session_id=([0-9a-f]{16})")


def manifest_session_ids(manifest: Path = MANIFEST) -> list[str]:
    if not manifest.is_file():
        return []
    return _SESSION_ID_RE.findall(manifest.read_text(encoding="utf-8"))


def main() -> int:
    ids = manifest_session_ids()
    if not ids:
        print(f"mark_holdout: no session_ids found in {MANIFEST}", file=sys.stderr)
        return 1

    from racingoptimizer.ingest import catalog as cat
    from racingoptimizer.ingest.api import catalog_path, resolve_corpus_root

    root = Path(resolve_corpus_root(None))
    db = catalog_path(root)
    if not db.is_file():
        print(f"mark_holdout: no catalog at {db} (run `optimize learn` first)",
              file=sys.stderr)
        return 1

    with cat.open_catalog(db) as conn:
        cat.set_held_out_sessions(conn, ids, held_out=True)
        rows = conn.execute(
            "SELECT session_id, held_out FROM sessions WHERE session_id IN "
            f"({','.join('?' * len(ids))})",
            ids,
        ).fetchall()
    present = {sid: int(flag) for sid, flag in rows}
    for sid in ids:
        state = present.get(sid)
        label = "FLAGGED" if state == 1 else ("NOT IN CATALOG" if state is None else "FAILED")
        print(f"  {sid}: {label}")
    print(f"mark_holdout: {sum(1 for v in present.values() if v == 1)}/{len(ids)} "
          "held-out sessions flagged gate-only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
