"""Stdlib-only GitHub Actions gates for finalized FPL Gameweeks."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from gameweeks import summarize, truthy as _truthy


def fixture_status(path: Path, season: str) -> dict[str, int | bool]:
    if not path.exists():
        return summarize([], season).status()
    with path.open(newline="", encoding="utf-8") as fh:
        return summarize(csv.DictReader(fh), season).status()


def last_replayed(path: Path, season: str) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    current = [r for r in rows if r.get("season") == season]
    try:
        return max((int(float(r["gw"])) for r in current if r.get("gw")), default=0)
    except (TypeError, ValueError):
        return 0


def _emit(values: dict[str, object]) -> None:
    lines = [f"{key}={str(value).lower() if isinstance(value, bool) else value}"
             for key, value in values.items()]
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    print(" ".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("season", "replay"))
    parser.add_argument("--season", default=os.environ.get("SEASON", "2026-2027"))
    parser.add_argument("--fixtures", type=Path, default=Path("data/fixtures.csv"))
    parser.add_argument("--replay", type=Path, default=Path("data/season_replay.csv"))
    args = parser.parse_args()

    status = fixture_status(args.fixtures, args.season)
    if args.mode == "season":
        _emit(status)
        return

    replayed = last_replayed(args.replay, args.season)
    force = _truthy(os.environ.get("FORCE", "false"))
    manual = os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"
    _emit({
        **status,
        "last_replayed": replayed,
        "should_run": force or manual or status["last_finalized"] > replayed,
    })


if __name__ == "__main__":
    main()
