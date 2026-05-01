"""Top-level `optimize` CLI group.

Composed by each slice that contributes commands:
- `learn` (slice A) — owned by `racingoptimizer.ingest.cli`
- `compare`, `status`, and the positional `<car> <track>` / `<ibt_path>`
  recommend invocations (slice F) — owned by `racingoptimizer.cli.recommend`

Positional shorthands routed to the `recommend` subcommand so the user
never has to type ``optimize recommend ...``:

* ``optimize <car> <track>`` — first arg is one of the canonical car keys.
* ``optimize <path/to/file.ibt>`` — first arg is an existing ``.ibt`` file;
  the recommend command then auto-detects car and track from the filename
  (VISION §8 "drop in an IBT, get a setup out").
"""
from __future__ import annotations

from pathlib import Path

import click

from racingoptimizer.cli.recommend import (
    CANONICAL_CARS,
    compare_cmd,
    recommend_cmd,
    status_cmd,
)
from racingoptimizer.ingest.cli import learn_command


class _OptimizeGroup(click.Group):
    """Click group dispatching positional shorthands to `recommend`."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if args:
            first = args[0]
            if first.strip().lower() in CANONICAL_CARS:
                args = ["recommend", *args]
            else:
                # Treat any existing .ibt file as an auto-detect recommend.
                candidate = Path(first)
                if candidate.suffix.lower() == ".ibt" and candidate.exists():
                    args = ["recommend", *args]
        return super().parse_args(ctx, args)


@click.command(cls=_OptimizeGroup, invoke_without_command=True)
@click.pass_context
def main(ctx: click.Context) -> None:
    """racingoptimizer CLI — `optimize <car> <track>` to recommend a setup."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


main.add_command(learn_command)
main.add_command(compare_cmd)
main.add_command(status_cmd)
main.add_command(recommend_cmd)


__all__ = ["main"]
