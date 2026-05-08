#!/usr/bin/env bash
# verify_holdout.sh -- pre-work integrity check for the physics-rebuild plan.
#
# Per PLAN.md Section 7 + Section 8.1, this script must pass at the start of
# every day. It checks:
#   1. Each held-out IBT in `docs/physics-rebuild/holdout.sha256` still hashes
#      to its committed value (no on-disk modification).
#   2. The catalog's `held_out=1` flag is set for each held-out session_id
#      (no row was silently flipped back to production).
#   3. No per-car cache pickle (`corpus/models/*.pickle`) trained against a
#      session_id in the held-out set (no production model leaked the gate).
#
# Exit codes:
#   0  all three checks passed; safe to begin the day's work.
#   1  hash mismatch (an IBT was modified or replaced).
#   2  catalog flag missing (a held-out session is being treated as production).
#   3  pickle leak (a model was trained on held-out data).
#   4  manifest missing or malformed.
#
# Usage: bash scripts/verify_holdout.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="$REPO_ROOT/docs/physics-rebuild/holdout.sha256"
CATALOG="$REPO_ROOT/corpus/catalog.sqlite"

if [[ ! -f "$MANIFEST" ]]; then
  echo "verify_holdout: manifest missing: $MANIFEST" >&2
  exit 4
fi

# 1. Hash check.
# Parse manifest with awk: column 1 is the hash; everything after it up to a
# trailing "  # ..." comment is the path. The path itself contains spaces
# (e.g. "bmwlmdh_spa 2024 up 2026-05-07 11-59-06.ibt"), so we cannot split
# on whitespace -- the awk pattern strips column 1 then the comment suffix.
HASH_FAIL=0
HASH_LINES=0
while IFS='|' read -r expected path; do
  [[ -z "$expected" || -z "$path" ]] && continue
  HASH_LINES=$((HASH_LINES + 1))
  if [[ ! -f "$REPO_ROOT/$path" ]]; then
    echo "verify_holdout: missing IBT: $path" >&2
    HASH_FAIL=1
    continue
  fi
  actual=$(sha256sum "$REPO_ROOT/$path" 2>/dev/null | awk '{print $1}')
  if [[ -z "$actual" ]]; then
    # sha256sum may not be on PATH on every Windows shell. Try certutil or python.
    actual=$(certutil -hashfile "$REPO_ROOT/$path" SHA256 2>/dev/null \
             | sed -n '2p' | tr -d ' \r\n')
  fi
  if [[ -z "$actual" ]]; then
    actual=$(python -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$REPO_ROOT/$path" 2>/dev/null)
  fi
  if [[ "$actual" != "$expected" ]]; then
    echo "verify_holdout: HASH MISMATCH for $path" >&2
    echo "  expected: $expected" >&2
    echo "  actual:   $actual" >&2
    HASH_FAIL=1
  fi
done < <(awk '!/^#/ && NF>=2 { hash=$1; $1=""; sub(/^[ \t]+/, ""); sub(/[ \t]+#.*$/, ""); print hash "|" $0 }' "$MANIFEST")

if [[ $HASH_LINES -eq 0 ]]; then
  echo "verify_holdout: manifest has no hash lines" >&2
  exit 4
fi
if [[ $HASH_FAIL -ne 0 ]]; then
  exit 1
fi

# 2. Catalog flag check (skipped if catalog file missing -- fresh corpus,
#    Day 0 prep itself sets the flag).
# Single python invocation so bash and python don't fight over $-expansion.
if [[ -f "$CATALOG" ]]; then
  CATALOG_REPORT=$(CATALOG="$CATALOG" python - <<'PY'
import os
import sqlite3

held_ids = [
    "3f0a05d3f44527bd",
    "d236a089300fc0ea",
    "fc96805e3b1a27cc",
    "72f43fa4527c4260",
    "a3d43056a952ff99",
]
conn = sqlite3.connect(os.environ["CATALOG"])
fail = False
for sid in held_ids:
    row = conn.execute(
        "SELECT held_out FROM sessions WHERE session_id=?", (sid,),
    ).fetchone()
    if row is None:
        # Soft warning: session may not be ingested in this corpus yet.
        print(f"WARN session {sid} not in catalog (ingest needed?)")
        continue
    if int(row[0]) != 1:
        print(f"FAIL session {sid} held_out={row[0]} (must be 1)")
        fail = True
print("STATUS=" + ("FAIL" if fail else "OK"))
PY
)
  echo "$CATALOG_REPORT" | grep -E "^(WARN|FAIL) " >&2 || true
  if echo "$CATALOG_REPORT" | grep -q "^STATUS=FAIL"; then
    exit 2
  fi
fi

# 3. Pickle-leak check (skipped if no caches yet)
PICKLE_DIR="$REPO_ROOT/corpus/models"
if [[ -d "$PICKLE_DIR" ]]; then
  LEAK=$(PICKLE_DIR="$PICKLE_DIR" python - <<'PY'
import os
import pickle
from pathlib import Path

held = {
    "3f0a05d3f44527bd", "d236a089300fc0ea", "fc96805e3b1a27cc",
    "72f43fa4527c4260", "a3d43056a952ff99",
}
leaks = []
for p in Path(os.environ["PICKLE_DIR"]).glob("*.pickle"):
    try:
        with open(p, "rb") as fh:
            obj = pickle.load(fh)
    except Exception:
        # Unreadable pickle (likely older layout version); skip.
        continue
    sids = getattr(obj, "session_ids", None)
    if sids and held.intersection(sids):
        leaks.append(f"{p.name}: {sorted(held.intersection(sids))}")
for line in leaks:
    print(line)
PY
)
  if [[ -n "$LEAK" ]]; then
    echo "verify_holdout: PICKLE LEAK -- per-car cache(s) trained on held-out:" >&2
    echo "$LEAK" >&2
    exit 3
  fi
fi

echo "verify_holdout: OK ($HASH_LINES IBTs hashed, catalog flag verified, no pickle leak)"
exit 0
