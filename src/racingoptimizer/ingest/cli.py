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
def learn_command(path: Path, corpus_root: Path | None) -> None:
    """Ingest a .ibt file or every .ibt under a directory into the corpus."""
    ids = _learn(path, corpus_root=corpus_root)
    click.echo(f"ingested {len(ids)} session(s)")
    for sid in ids:
        click.echo(f"  {sid}")
