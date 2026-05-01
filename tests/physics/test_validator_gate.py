"""Pytest replication of the `physics-fit-validator` subagent's three core
checks (spec §13, gap #19).

The subagent at `.claude/agents/physics-fit-validator.md` is the human-loop
gate that VISION §3/§6 promises will block any leaking-formula or sparse-
overconfident fit before merge. Until a proper CI hook invokes the subagent
directly, these three pytest checks replicate its workflow as code so the
gate runs every time `uv run pytest` does.

Three checks per the worker brief:

1. ``test_no_lap_time_in_objective`` — grep `score.py` and `recommend.py`
   for `lap_time` / `laptime` (case-insensitive); both must be absent.
2. ``test_no_textbook_formulas_in_score`` — heuristic regex against
   `score.py` looking for hardcoded magic numbers like `* Speed`, `Speed^2`,
   etc.; per-car-baseline references (`baselines.*`, `model.baseline_setup`,
   self.baseline_setup) are allowed.
3. ``test_confidence_aligns_with_data_density`` — fit a tiny model from a
   single BMW Sebring lap; assert every fitter with `n_samples < 30`
   reports `regime == "sparse"` and that bracket width grows with
   `cv_residual_std`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from racingoptimizer.confidence import Confidence
from racingoptimizer.ingest.api import learn, sessions
from racingoptimizer.physics import fit
from racingoptimizer.track import build_track_model

REPO_ROOT = Path(__file__).resolve().parents[2]
PHYSICS_DIR = REPO_ROOT / "src" / "racingoptimizer" / "physics"
_IBT_DIR = REPO_ROOT / "ibtfiles"
BMW_SEBRING_IBT = _IBT_DIR / "bmwlmdh_sebring international 2026-03-22 14-52-24.ibt"


# --- Check 1: lap-time absence -------------------------------------------


def _strip_comments_and_docstrings(text: str) -> str:
    """Remove `#` line comments and triple-quoted strings.

    The grep below only cares about *executable* references, so comments
    and docstrings (which legitimately discuss the prohibition) are stripped.
    """
    # Remove triple-quoted strings (greedy across lines).
    text = re.sub(r'"""[\s\S]*?"""', "", text)
    text = re.sub(r"'''[\s\S]*?'''", "", text)
    # Remove `#` line comments.
    out_lines: list[str] = []
    for line in text.splitlines():
        idx = line.find("#")
        if idx >= 0:
            line = line[:idx]
        out_lines.append(line)
    return "\n".join(out_lines)


@pytest.mark.parametrize("module_name", ["score.py", "recommend.py"])
def test_no_lap_time_in_objective(module_name: str) -> None:
    """Spec §6 hard rule: lap time NEVER appears in the optimisation objective."""
    path = PHYSICS_DIR / module_name
    raw = path.read_text(encoding="utf-8")
    code = _strip_comments_and_docstrings(raw).lower()
    assert "lap_time" not in code, (
        f"{module_name} references `lap_time` outside comments/docstrings — "
        f"spec §6 forbids lap time inside the optimisation objective"
    )
    assert "laptime" not in code, (
        f"{module_name} references `laptime` outside comments/docstrings — "
        f"spec §6 forbids lap time inside the optimisation objective"
    )


# --- Check 2: textbook-formula leakage -----------------------------------


# Patterns that signal a hardcoded engineering equation hiding inside the
# scorer. The empirical-only rule (CLAUDE.md, spec §3) means score.py must
# not multiply raw channels by literal coefficients — every relationship
# should come from a fitted PhysicsModel.
_FORBIDDEN_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bSpeed\s*\*\*\s*2", "Speed**2 — looks like a textbook drag/downforce term"),
    (r"\*\s*Speed\b", "* Speed — looks like a textbook coefficient*velocity term"),
    (
        r"\bAccelLat\s*\*\s*\d+(?:\.\d+)?",
        "AccelLat * <literal> — looks like a hardcoded G-force coefficient",
    ),
    (
        r"\bspring[_-]?rate\s*\*\s*\d+(?:\.\d+)?",
        "spring_rate * <literal> — looks like a textbook K*x term",
    ),
)


def test_no_textbook_formulas_in_score() -> None:
    """Heuristic guard against `f = k*x` style leakage (spec §3, §13)."""
    path = PHYSICS_DIR / "score.py"
    raw = path.read_text(encoding="utf-8")
    code = _strip_comments_and_docstrings(raw)

    flagged: list[str] = []
    for pattern, description in _FORBIDDEN_PATTERNS:
        match = re.search(pattern, code)
        if match is None:
            continue
        # Whitelist: the match sits on a line that explicitly references the
        # per-car baseline (model.baseline_setup / self.baseline_setup /
        # baselines.*) — those are empirically-grounded falls-through, not
        # textbook formulas.
        line_start = code.rfind("\n", 0, match.start()) + 1
        line_end = code.find("\n", match.end())
        if line_end < 0:
            line_end = len(code)
        line = code[line_start:line_end]
        if any(token in line for token in ("baseline_setup", "baselines.")):
            continue
        flagged.append(f"{pattern!r} ({description}) — line: {line.strip()!r}")

    assert not flagged, (
        "score.py contains suspected textbook formulas:\n  - "
        + "\n  - ".join(flagged)
        + "\nIf these are intentional empirical lookups (e.g. an aero map "
        "call), refactor to make the data-source explicit."
    )


# --- Check 3: confidence-vs-data-density alignment ------------------------


@pytest.fixture(scope="module")
def cold_start_model(tmp_path_factory):
    """Fit a one-session BMW Sebring model (cold-start, sparse-by-design)."""
    from tests._lfs_util import is_unmaterialised_lfs_pointer, lfs_skip_message

    if not BMW_SEBRING_IBT.exists():
        pytest.skip(f"missing BMW Sebring fixture at {BMW_SEBRING_IBT}")
    if is_unmaterialised_lfs_pointer(BMW_SEBRING_IBT):
        pytest.skip(lfs_skip_message(BMW_SEBRING_IBT))
    root = tmp_path_factory.mktemp("validator_gate_corpus") / "corpus"
    root.mkdir()
    sids = learn(BMW_SEBRING_IBT, corpus_root=root)
    sess_df = sessions(corpus_root=root)
    car = sess_df.row(0, named=True)["car"]
    track = sess_df.row(0, named=True)["track"]
    tm = build_track_model(track, sids, corpus_root=root)
    return fit(car, sids, tm, corpus_root=root, k_folds=2, seed=0xC0FFEE)


def test_confidence_aligns_with_data_density(cold_start_model) -> None:
    """Sparse-by-sample-count fitters must report `regime == "sparse"`.

    Also asserts the fundamental Confidence-bracket invariant: brackets
    derived from a larger ``cv_residual_std`` are at least as wide as
    brackets derived from a smaller one (monotonic in residual std).
    """
    model = cold_start_model
    # 1. Every fitter with n_samples < 30 must be tagged sparse.
    sparse_ok = 0
    sparse_violations: list[str] = []
    for key, record in model.fitters.items():
        if record.n_samples >= 30:
            continue
        conf = Confidence.derive(
            value=0.0,
            n_samples=int(record.n_samples),
            cv_residual_std=float(record.cv_residual_std),
            signal_std=float(max(record.signal_std, 1e-12)),
        )
        if conf.regime == "sparse":
            sparse_ok += 1
        else:
            sparse_violations.append(
                f"{key} n_samples={record.n_samples} regime={conf.regime!r}"
            )
    assert not sparse_violations, (
        "fitters with <30 samples leaked a non-sparse regime — Confidence."
        "derive's sparse short-circuit is broken:\n  "
        + "\n  ".join(sparse_violations)
    )
    assert sparse_ok > 0, (
        "the BMW Sebring single-session corpus is expected to produce at "
        "least one sparse fitter; got zero — fixture changed?"
    )

    # 2. Bracket width is monotonic in cv_residual_std (sanity check that
    #    Confidence.derive is the single bracket-width source of truth).
    a = Confidence.derive(value=0.0, n_samples=50, cv_residual_std=0.1, signal_std=1.0)
    b = Confidence.derive(value=0.0, n_samples=50, cv_residual_std=0.5, signal_std=1.0)
    assert (b.hi - b.lo) > (a.hi - a.lo), (
        "Confidence bracket width must grow with cv_residual_std — got "
        f"a.width={a.hi - a.lo} >= b.width={b.hi - b.lo}"
    )
