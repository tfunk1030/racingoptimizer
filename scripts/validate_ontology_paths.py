"""Validate per-car ontology JSON paths against real IBT CarSetup blobs.

For each of the five canonical GTP fixtures (one IBT per car), this script:

1. Reads the embedded ``CarSetup`` YAML block.
2. Walks every entry in ``ontology_for(car)`` and reports whether the
   ontology's ``json_path`` resolves to a real value, the raw value,
   the coerced numeric value, and the parameter's flags.
3. Inventories the YAML via :func:`racingoptimizer.physics.garage_inventory.inventory_setup`
   and surfaces every leaf classified ``blocked_user_input`` that has no
   ontology entry (potential coverage gap).
4. Cross-checks that every parameter returned by
   :func:`fittable_parameters` resolves to a non-None scalar (the same
   invariant ``tests/physics/test_ontology_per_car.py`` enforces).

Output is a Markdown report written to ``docs/PARAMETER_VALIDATION.md``
plus a one-line summary per car on stdout.

Run from the repo root with materialised LFS fixtures::

    git lfs pull --include='ibtfiles/<canonical fixtures>'
    uv run python scripts/validate_ontology_paths.py

Cars whose canonical fixture is still a git-LFS pointer are skipped with
an explicit note in the report; the script does not fail in that case.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from racingoptimizer.constraints import load_constraints  # noqa: E402
from racingoptimizer.physics.garage_inventory import inventory_setup  # noqa: E402
from racingoptimizer.physics.ontology import (  # noqa: E402
    fittable_parameters,
    ontology_for,
    setup_value,
)

IBT_DIR = REPO_ROOT / "ibtfiles"
REPORT_PATH = REPO_ROOT / "docs" / "PARAMETER_VALIDATION.md"

PER_CAR_FIXTURES: dict[str, str] = {
    "acura": "acuraarx06gtp_daytona 2011 road 2026-04-03 20-31-55.ibt",
    "bmw": "bmwlmdh_sebring international 2026-03-22 14-52-24.ibt",
    "cadillac": "cadillacvseriesrgtp_lagunaseca 2026-04-27 19-50-46.ibt",
    "ferrari": "ferrari499p_hockenheim gp 2026-03-31 15-49-42.ibt",
    "porsche": "porsche963gtp_algarve gp 2026-04-07 15-49-17.ibt",
}

_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def _is_lfs_pointer(path: Path) -> bool:
    if not path.exists() or path.stat().st_size > 4096:
        return False
    try:
        with path.open("rb") as fh:
            return fh.read(len(_LFS_POINTER_PREFIX)) == _LFS_POINTER_PREFIX
    except OSError:
        return False


def _read_setup_yaml(ibt_path: Path) -> dict:
    try:
        import irsdk
    except ImportError:
        import pyirsdk as irsdk  # type: ignore[import-not-found, no-redef]

    from racingoptimizer.ingest.parser import _read_yaml

    ibt = irsdk.IBT()
    ibt.open(str(ibt_path))
    try:
        info = _read_yaml(ibt)
    finally:
        ibt.close()
    setup = info.get("CarSetup", {}) or {}
    if not isinstance(setup, dict):
        raise RuntimeError(f"unexpected CarSetup shape in {ibt_path}: {type(setup)}")
    return setup


@dataclass(frozen=True)
class ParamRow:
    name: str
    json_path: tuple[str, ...]
    units: str
    family: str
    fittable: bool
    user_settable: bool
    raw_value: object
    coerced: float | None
    status: str  # OK | MISMATCH | BLOCKED | READOUT


@dataclass(frozen=True)
class CarReport:
    car: str
    skipped: str | None
    rows: list[ParamRow]
    fittable_failures: list[tuple[str, tuple[str, ...]]]
    unmapped_blocked: list[tuple[tuple[str, ...], object, str]]
    unmapped_readouts: list[tuple[tuple[str, ...], object]]


def _walk_raw(setup: dict, path: tuple[str, ...]) -> object:
    cur: object = setup
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _classify_row(spec, raw, coerced, fittable_set: set[str], name: str) -> str:
    if not spec.fittable:
        return "READOUT" if not spec.user_settable else "BLOCKED"
    if name not in fittable_set:
        return "BLOCKED"
    if raw is None:
        return "MISMATCH"
    if coerced is None:
        return "MISMATCH"
    return "OK"


def validate_car(car: str) -> CarReport:
    fixture = IBT_DIR / PER_CAR_FIXTURES[car]
    if not fixture.exists():
        return CarReport(
            car=car,
            skipped=f"fixture missing on disk: {fixture}",
            rows=[],
            fittable_failures=[],
            unmapped_blocked=[],
            unmapped_readouts=[],
        )
    if _is_lfs_pointer(fixture):
        msg = (
            "fixture is a git-LFS pointer "
            f"(run `git lfs pull --include='ibtfiles/{PER_CAR_FIXTURES[car]}'`)"
        )
        return CarReport(
            car=car,
            skipped=msg,
            rows=[],
            fittable_failures=[],
            unmapped_blocked=[],
            unmapped_readouts=[],
        )

    setup = _read_setup_yaml(fixture)
    onto = ontology_for(car)
    constraints = load_constraints()
    fittable_set = set(fittable_parameters(car, constraints))

    rows: list[ParamRow] = []
    for name, spec in sorted(onto.items()):
        raw = _walk_raw(setup, spec.json_path)
        coerced = setup_value(car, name, setup)
        rows.append(
            ParamRow(
                name=name,
                json_path=spec.json_path,
                units=spec.units,
                family=spec.family,
                fittable=spec.fittable,
                user_settable=spec.user_settable,
                raw_value=raw,
                coerced=coerced,
                status=_classify_row(spec, raw, coerced, fittable_set, name),
            )
        )

    fittable_failures = [
        (r.name, r.json_path) for r in rows if r.name in fittable_set and r.coerced is None
    ]

    inventory = inventory_setup(car, setup, constraints)
    onto_paths = {spec.json_path for spec in onto.values()}
    unmapped_blocked: list[tuple[tuple[str, ...], object, str]] = []
    unmapped_readouts: list[tuple[tuple[str, ...], object]] = []
    for leaf in inventory:
        if leaf.path in onto_paths:
            continue
        if leaf.classification == "blocked_user_input":
            unmapped_blocked.append((leaf.path, leaf.value, leaf.reason))
        elif leaf.classification == "unsupported_readout":
            unmapped_readouts.append((leaf.path, leaf.value))

    return CarReport(
        car=car,
        skipped=None,
        rows=rows,
        fittable_failures=fittable_failures,
        unmapped_blocked=unmapped_blocked,
        unmapped_readouts=unmapped_readouts,
    )


def _format_path(path: tuple[str, ...]) -> str:
    return ".".join(path) if path else "(root)"


def _format_value(v: object) -> str:
    if v is None:
        return "—"
    if isinstance(v, dict):
        return f"<dict: {sorted(v.keys())[:3]}…>"
    return repr(v)


def _markdown_summary(reports: list[CarReport]) -> list[str]:
    lines: list[str] = ["## Summary", ""]
    lines.append(
        "| car | skipped | params | OK | MISMATCH | BLOCKED | READOUT "
        "| unmapped (blocked) | unmapped (readout) |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in reports:
        if r.skipped:
            lines.append(f"| {r.car} | {r.skipped} | — | — | — | — | — | — | — |")
            continue
        ok = sum(1 for x in r.rows if x.status == "OK")
        mismatch = sum(1 for x in r.rows if x.status == "MISMATCH")
        blocked = sum(1 for x in r.rows if x.status == "BLOCKED")
        readout = sum(1 for x in r.rows if x.status == "READOUT")
        lines.append(
            f"| {r.car} | — | {len(r.rows)} | {ok} | {mismatch} | {blocked} | {readout} | "
            f"{len(r.unmapped_blocked)} | {len(r.unmapped_readouts)} |"
        )
    lines.append("")
    return lines


def _markdown_per_car(report: CarReport) -> list[str]:
    lines = [f"## {report.car}", ""]
    if report.skipped:
        lines.append(f"_Skipped: {report.skipped}_")
        lines.append("")
        return lines

    lines.append(f"Fixture: `ibtfiles/{PER_CAR_FIXTURES[report.car]}`")
    lines.append("")
    lines.append("### Ontology parameters")
    lines.append("")
    lines.append(
        "| parameter | json_path | units | family | fittable | user_settable "
        "| raw value | coerced | status |"
    )
    lines.append("| --- | --- | --- | --- | :-: | :-: | --- | ---: | --- |")
    for row in report.rows:
        lines.append(
            f"| `{row.name}` "
            f"| `{_format_path(row.json_path)}` "
            f"| {row.units} "
            f"| {row.family} "
            f"| {'✓' if row.fittable else '·'} "
            f"| {'✓' if row.user_settable else '·'} "
            f"| {_format_value(row.raw_value)} "
            f"| {'' if row.coerced is None else f'{row.coerced:g}'} "
            f"| {row.status} |"
        )
    lines.append("")

    if report.fittable_failures:
        lines.append("### Fittable failures (mapped but unresolved)")
        lines.append("")
        for name, path in report.fittable_failures:
            lines.append(f"- `{name}` → `{_format_path(path)}`")
        lines.append("")
    else:
        lines.append(
            "_No fittable failures: every fittable parameter "
            "resolves to a numeric value._"
        )
        lines.append("")

    if report.unmapped_blocked:
        lines.append("### Unmapped user-settable leaves (potential coverage gaps)")
        lines.append("")
        lines.append("| path | value | reason |")
        lines.append("| --- | --- | --- |")
        for path, value, reason in sorted(report.unmapped_blocked):
            lines.append(f"| `{_format_path(path)}` | {_format_value(value)} | {reason} |")
        lines.append("")
    else:
        lines.append(
            "_No unmapped user-settable leaves: every garage input has an ontology "
            "entry or is explicitly blocked._"
        )
        lines.append("")

    if report.unmapped_readouts:
        n = len(report.unmapped_readouts)
        lines.append(f"<details><summary>Unsupported readouts ({n} leaves)</summary>")
        lines.append("")
        for path, value in sorted(report.unmapped_readouts):
            lines.append(f"- `{_format_path(path)}` = {_format_value(value)}")
        lines.append("")
        lines.append("</details>")
        lines.append("")
    return lines


def _markdown_finding_zero() -> list[str]:
    return [
        "## Finding 0 — `brake_bias_pct` was NOT renamed to `brake_duct_*`",
        "",
        "These are two different parameters that live in different parts of the codebase:",
        "",
        "| concept | ontology entry | family | constraints.md section | bounds status |",
        "| --- | --- | --- | --- | --- |",
        (
            "| brake bias | `brake_bias_pct` (key in `ontology.py`) | `brake_bias` "
            "| `### Brake bias` (constraints.md:173-179) | bounded 40–60 % |"
        ),
        (
            "| brake duct (front/rear) | _not in ontology_ | _n/a_ "
            "| `### Brake duct opening — front/rear` (constraints.md:210-218) "
            "| TODO bounds (from iRacing UI) |"
        ),
        "",
        "What changed in commit `a407f0b`:",
        "",
        (
            "* `src/racingoptimizer/cli/recommend.py:1045-1049` updated a status-note string. "
            "`brake_bias` was REMOVED from the \"missing-bounds\" list (because bounds were "
            "added in `bf2e48b`); `brake_ducts` REMAINS in that list (still TODO)."
        ),
        (
            "* `tests/physics/test_ce_degradation.py` flipped "
            "`assert \"brake_bias_pct\" in untrained` → "
            "`assert \"brake_bias_pct\" not in untrained` "
            "(brake-bias is now trained, not CE-gated)."
        ),
        (
            "* The ontology entry name and family are unchanged: "
            "`brake_bias_pct`, `family=\"brake_bias\"`. The default JSON path remains "
            "`BrakesDriveUnit.BrakeSpec.BrakePressureBias`; per-car overrides for Acura and "
            "Ferrari rewire it to `Systems.BrakeSpec.BrakePressureBias` "
            "(validated by the per-car tables above)."
        ),
        "",
        (
            "There is no ontology entry called `brake_duct_*`; the only `brake_duct` "
            "references in the source are the constraints-loader header parser "
            "(`src/racingoptimizer/constraints/loader.py:145-147`) and the TODO bounds "
            "tables in `constraints.md`."
        ),
        "",
    ]


def main() -> int:
    reports = [validate_car(car) for car in sorted(PER_CAR_FIXTURES)]
    all_skipped = all(r.skipped for r in reports)

    lines: list[str] = [
        "# Per-car parameter validation report",
        "",
        "Generated by `scripts/validate_ontology_paths.py`. For every car in"
        " the optimizer's ontology, this report cross-references the JSON paths in"
        " `src/racingoptimizer/physics/ontology.py` against the canonical IBT"
        " fixture's embedded `CarSetup` YAML.",
        "",
    ]
    if all_skipped:
        include_paths = ",".join(
            f"ibtfiles/{f}" for f in PER_CAR_FIXTURES.values()
        )
        lines.extend([
            (
                "> **Notice — empirical tables are empty in this commit.** Every "
                "canonical IBT fixture in this checkout is still a git-LFS pointer "
                "(the sandbox that generated this report could not reach the LFS "
                "endpoint). To populate the per-car tables, run from a checkout "
                "that has materialised LFS objects:"
            ),
            "",
            "```bash",
            "git lfs install",
            f"git lfs pull --include='{include_paths}'",
            "uv run python scripts/validate_ontology_paths.py",
            "```",
            "",
            "Re-running the script overwrites this file with the populated report.",
            "",
        ])
    lines.extend([
        "Status legend:",
        "",
        "* **OK** — fittable parameter, ontology path resolves to a numeric value.",
        (
            "* **MISMATCH** — fittable parameter, ontology path does NOT resolve. "
            "Likely wrong path."
        ),
        (
            "* **BLOCKED** — known user input, fitting/recommendation deliberately "
            "disabled (e.g. `_blocked_like` entries, untrained dampers)."
        ),
        (
            "* **READOUT** — calculated readout (`user_settable=False`); the optimizer "
            "learns it as a target but never recommends it."
        ),
        "",
    ])
    lines.extend(_markdown_finding_zero())
    lines.extend(_markdown_summary(reports))
    for report in reports:
        lines.extend(_markdown_per_car(report))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n")

    print(f"Wrote {REPORT_PATH.relative_to(REPO_ROOT)}")
    failure = False
    for r in reports:
        if r.skipped:
            print(f"  {r.car}: SKIPPED — {r.skipped}")
            continue
        ok = sum(1 for x in r.rows if x.status == "OK")
        mismatch = sum(1 for x in r.rows if x.status == "MISMATCH")
        print(
            f"  {r.car}: {ok} OK / {mismatch} MISMATCH / "
            f"{len(r.unmapped_blocked)} unmapped-blocked"
        )
        if r.fittable_failures:
            failure = True
            for name, path in r.fittable_failures:
                print(f"    !! {name} -> {_format_path(path)}")
    return 1 if failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
