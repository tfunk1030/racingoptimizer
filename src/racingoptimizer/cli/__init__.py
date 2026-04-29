"""Top-level `optimize` CLI group.

Composed by each slice that contributes commands:
- `learn` (slice A) — owned by `racingoptimizer.ingest.cli`
- `compare`, `status`, and the positional `<car> <track>` recommend
  invocation (slice F) — owned by `racingoptimizer.cli.recommend`

Positional shorthand: `optimize <car> <track>` is auto-routed to the
`recommend` subcommand when the first argument is one of the canonical
car keys, so the user never has to type `optimize recommend ...`.
"""
from __future__ import annotations

import click

from racingoptimizer.cli.recommend import (
    CANONICAL_CARS,
    compare_cmd,
    recommend_cmd,
    status_cmd,
)
from racingoptimizer.ingest.cli import learn_command


class _OptimizeGroup(click.Group):
    """Click group that dispatches `optimize <car> <track>` to `recommend`."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if args and args[0].strip().lower() in CANONICAL_CARS:
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
