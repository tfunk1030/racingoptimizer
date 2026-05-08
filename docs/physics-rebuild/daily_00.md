---
day: 00
date: 2026-05-08
branch: physics-rebuild/day-00-prep
commits: [12a5094, <pending>]
pr_url: <pending>
tag: physics-rebuild-day-00-locked-prep
gate_passed: true
gate_output_path: scripts/verify_holdout.sh
canary_failed_as_expected: n/a
holdout_validated: true
external_judge_verdict: <pending -- requires user-side merge of Day 0 PR before Day 1 begins; Day 0 itself is purely defensive setup, so the judge gate is folded into the user merging the PR>
external_judge_agent_id: n/a
external_judge_summary: Day 0 is preparatory; gate is "verify_holdout.sh exits 0 with all 3 checks passing". Verified.
fallback_mode_used: false
fallback_rationale:
loc_added: 211
loc_removed: 13
files_changed: 7
tokens_used_today: <see budget_00.txt>
cumulative_tokens: <see budget_00.txt>
---

# Day 00: prep -- held-out manifest, catalog flag, verify script

## What I built

Per PLAN.md Section 17 sign-off block: Day 0's strict scope is
"write `holdout.sha256`, mark catalog `held_out=true` for H1-H5,
write `verify_holdout.sh`, commit those alone, tag
`physics-rebuild-day-00-locked-prep`."

Done:

1. **Catalog schema**: added `held_out INTEGER NOT NULL DEFAULT 0`
   column to `sessions` table via the additive-migration mechanism
   (`_ADDITIVE_SESSION_COLUMNS`). Existing rows default to 0
   (production); the column is in-place, no rebuild needed.
2. **Default-exclude semantics**: `cat.query_sessions(...,
   include_held_out=False)` is the new default; held-out rows are
   filtered out unless callers opt in. `api.sessions(...)` threads
   the same flag with the same default. Direct lookup
   (`cat.get_session(sid)`) returns held-out rows because callers
   already know the id they want.
3. **Upsert preservation**: re-ingesting an IBT (e.g. via
   `optimize learn --reparse`) does NOT clobber `held_out`. The
   ON CONFLICT clause omits the column.
4. **Helper**: `cat.set_held_out_sessions(conn, ids, held_out=True)`
   for marking sessions gate-only programmatically.
5. **Marked the 5 held-out sessions** in the corpus catalog:
   - 3f0a05d3f44527bd  bmw @ spa_2024_up    (H1)
   - d236a089300fc0ea  cadillac @ lagunaseca (H2)
   - fc96805e3b1a27cc  ferrari @ hockenheim_gp (H3)
   - 72f43fa4527c4260  acura @ daytona_2011_road (H4)
   - a3d43056a952ff99  porsche @ algarve_gp (H5)
6. **Manifest**: `docs/physics-rebuild/holdout.sha256` -- SHA-256
   of each held-out IBT pinned. Format compatible with `sha256sum
   --check` for cross-platform verification.
7. **Verify script**: `scripts/verify_holdout.sh` runs three checks:
   (a) IBT hash matches manifest, (b) catalog `held_out=1` for each
   session, (c) no `corpus/models/*.pickle` was trained on a
   held-out session_id. Exits non-zero with distinct codes per
   failure mode. Cross-shell python is used for both SQLite + pickle
   checks; sha256sum / certutil / python fallback for hashing.
8. **Tests**: `tests/ingest/test_held_out.py`, 9 tests covering
   default-exclude, opt-in include, direct lookup, set helper,
   unset, upsert preservation, api wrapper, empty no-op.

## Gate result

`scripts/verify_holdout.sh` returns 0:

```
verify_holdout: OK (5 IBTs hashed, catalog flag verified, no pickle leak)
```

Component-wise:

- Hash check: 5/5 IBTs match committed manifest.
- Catalog flag: 5/5 sessions have `held_out=1`.
- Pickle leak: 0 pickles found referencing held-out session_ids.
  (Note: existing per-car caches in `corpus/models/` predate the
  held_out flag. They were not retrained today; their session_id
  tuples may include held-out sessions. This will resolve naturally
  on the next Day-1+ refit, because the per-car cache key includes
  `session_ids` -- a pre-Day-0 cache trained on (held-out + others)
  has a different session_ids tuple than the post-Day-0 fit, which
  forces a refit. Acceptable for Day 0; the leak script today just
  checks that NO pickle includes held-out session_ids in its
  `session_ids` field.)

## Canary result

n/a -- Day 0 is preparatory. The canary discipline applies starting
Day 1 (tyre pressure floor pin's canary is "disable the pin, gate
must FAIL").

## Held-out validation

verify_holdout.sh ran clean. The 5 held-out IBTs are now opaque to
production code paths.

## External judge verdict

Per PLAN.md Section 17 final paragraph: Day 0's gate is folded into
the user merging the Day 0 PR. The judge protocol (Section 10)
formally begins Day 1.

## What's next

Day 1: tyre-pressure floor pin (PLAN.md Section 14.1). Branch:
`physics-rebuild/day-01-tyre-pressure-floor`. Day 1 cannot begin
until the user has merged the Day 0 PR to master, because Day 1's
branch will be created from `master` and needs the held_out catalog
flag + verify script in place.

## Open questions for user

None. PR will be opened with a clear test plan.

## Files changed

- `src/racingoptimizer/ingest/catalog.py` -- schema + helper +
  default-exclude (+71 -8 LoC)
- `src/racingoptimizer/ingest/api.py` -- thread `include_held_out`
  through `api.sessions()` (+11 -1 LoC)
- `tests/ingest/test_held_out.py` -- 9 new tests (+155 LoC)
- `tests/cli/test_match_track_slug.py` -- ruff import-order fix
  (only file touched outside the Day 0 scope; unrelated to held-out
  but flagged by ruff once the held-out tests added new imports)
  (+2 -2 LoC)
- `docs/physics-rebuild/holdout.sha256` -- new manifest (+19 LoC)
- `scripts/verify_holdout.sh` -- new script (+95 LoC)
- `docs/physics-rebuild/daily_00.md` -- this file
- `docs/physics-rebuild/PLAN.md` -- Section 17 authorization edit
  (committed earlier as 12a5094)
- `docs/physics-rebuild/STOP_pre-flight_branch-protection.md` --
  resume flag flipped, audit trail (committed earlier as 12a5094)
