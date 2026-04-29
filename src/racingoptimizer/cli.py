"""Top-level `optimize` CLI group."""
from __future__ import annotations

import click

from racingoptimizer.ingest.cli import learn_command


@click.group()
def main() -> None:
    """racingoptimizer CLI."""


main.add_command(learn_command)
