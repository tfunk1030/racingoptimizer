"""P3.3 -- thin-corpus refusal banner from `optimize <car> <track>`.

The CLI must refuse to emit a DE-driven race setup when the per-car
corpus is below ``_THIN_CORPUS_REFUSAL_N`` production sessions OR
when the per-axle grip ceiling fit failed (``axle_grip_ceilings is
None``). Both conditions cause hybrid scoring to silently collapse to
surrogate-only and the user would otherwise see a "physics-based"
briefing with no physics anchor.

Tests here exercise the refusal predicate + banner directly so we
don't need to fit a real per-car model. The CLI integration is one
``if`` branch in ``recommend_cmd``; covering the predicate is the
load-bearing piece.
"""
from __future__ import annotations

import json

import polars as pl

from racingoptimizer.cli.recommend import (
    _THIN_CORPUS_REFUSAL_N,
    _emit_thin_corpus_refusal,
    _is_thin_corpus_for_recommend,
    _thin_corpus_refusal_lines,
)


class _Model:
    def __init__(self, axle_grip_ceilings=None):
        self.axle_grip_ceilings = axle_grip_ceilings


def _sessions(n: int) -> pl.DataFrame:
    return pl.DataFrame({"session_id": [f"s{i}" for i in range(n)]})


def test_thin_session_count_triggers_refusal() -> None:
    model = _Model(axle_grip_ceilings={"front": object(), "rear": object()})
    df = _sessions(_THIN_CORPUS_REFUSAL_N - 1)
    assert _is_thin_corpus_for_recommend(model, df) is True


def test_session_count_at_threshold_with_axles_does_not_refuse() -> None:
    model = _Model(axle_grip_ceilings={"front": object(), "rear": object()})
    df = _sessions(_THIN_CORPUS_REFUSAL_N)
    assert _is_thin_corpus_for_recommend(model, df) is False


def test_missing_axle_ceilings_triggers_refusal_even_when_corpus_is_large() -> None:
    model = _Model(axle_grip_ceilings=None)
    df = _sessions(_THIN_CORPUS_REFUSAL_N + 50)
    assert _is_thin_corpus_for_recommend(model, df) is True


def test_refusal_lines_mention_calibrate_command_and_session_count() -> None:
    model = _Model(axle_grip_ceilings=None)
    lines = _thin_corpus_refusal_lines("acura", "belleisle", model, 13)
    joined = "\n".join(lines)
    assert "Corpus too thin" in joined
    assert "13" in joined
    assert "no axle ceilings fit" in joined
    assert "optimize calibrate acura belleisle" in joined


def test_refusal_lines_when_axle_ceilings_present_say_so() -> None:
    """If the corpus is below threshold but ceilings DID fit, the banner
    should not falsely claim "no axle ceilings fit"."""
    model = _Model(axle_grip_ceilings={"front": object(), "rear": object()})
    lines = _thin_corpus_refusal_lines("acura", "belleisle", model, 18)
    joined = "\n".join(lines)
    assert "axle ceilings fit" in joined
    assert "no axle ceilings fit" not in joined


def test_emit_thin_corpus_refusal_text_mode_prints_banner(capsys) -> None:
    model = _Model(axle_grip_ceilings=None)
    df = _sessions(13)
    _emit_thin_corpus_refusal(
        car_key="acura",
        track_slug="belleisle",
        model=model,
        catalog_sessions=df,
        as_json=False,
        output_file=None,
    )
    out = capsys.readouterr().out
    assert "Corpus too thin" in out
    assert "optimize calibrate acura belleisle" in out


def test_emit_thin_corpus_refusal_json_mode_emits_machine_readable(capsys) -> None:
    model = _Model(axle_grip_ceilings=None)
    df = _sessions(13)
    _emit_thin_corpus_refusal(
        car_key="acura",
        track_slug="belleisle",
        model=model,
        catalog_sessions=df,
        as_json=True,
        output_file=None,
    )
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["refused"] is True
    assert payload["reason"] == "thin_corpus"
    assert payload["car"] == "acura"
    assert payload["track"] == "belleisle"
    assert payload["n_production_sessions"] == 13
    assert payload["axle_grip_ceilings_present"] is False
    assert any("Corpus too thin" in w for w in payload["warnings"])
