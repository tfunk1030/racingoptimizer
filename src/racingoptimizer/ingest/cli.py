"""`optimize learn` subcommand."""
from __future__ import annotations

from pathlib import Path

import click

from racingoptimizer.ingest.api import learn as _learn


@click.command(name="learn")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--corpus-root",
    type=click.Path(path_type=Path),
    default=None,
    help="Override corpus location (else uses RACINGOPTIMIZER_CORPUS or repo default).",
)
@click.option(
    "--reparse",
    is_flag=True,
    default=False,
    help=(
        "Re-parse already-ingested sessions instead of short-circuiting on "
        "their cached `status=ok`. Use after a parser change (e.g. the "
        "filename-derived `recorded_at` fix) to refresh stale catalog "
        "fields without manual catalog surgery."
    ),
)
@click.option(
    "--no-quality-masks",
    "no_quality_masks",
    is_flag=True,
    default=False,
    help=(
        "Skip the slice-D track-quality mask pass after ingest. By default, "
        "every newly-ingested session has its `data_quality_mask` column "
        "rewritten using a TrackModel built from every session on the "
        "same (car, track), so the fitter masks out curb/off-track samples. "
        "Disabling this leaves the all-True default in place; use only "
        "when iterating quickly on the parser and you don't care about "
        "fit-quality output."
    ),
)
def learn_command(
    path: Path, corpus_root: Path | None, reparse: bool, no_quality_masks: bool,
) -> None:
    """Ingest a .ibt file or every .ibt under a directory into the corpus."""
    ids = _learn(
        path,
        corpus_root=corpus_root,
        reparse=reparse,
        apply_quality_masks=not no_quality_masks,
    )
    click.echo(f"ingested {len(ids)} session(s)")
    for sid in ids:
        click.echo(f"  {sid}")
