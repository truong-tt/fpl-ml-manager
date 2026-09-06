"""Gameweek completion and official finalization, independent of storage and ML."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "1.0", "yes"}


@dataclass(frozen=True)
class Gameweeks:
    completed: tuple[int, ...]
    finalized: tuple[int, ...]

    @property
    def last_completed(self) -> int:
        return max(self.completed, default=0)

    @property
    def last_finalized(self) -> int:
        return max(self.finalized, default=0)

    def status(self) -> dict[str, int | bool]:
        return {
            "last_completed": self.last_completed,
            "last_finalized": self.last_finalized,
            "review_pending": self.last_completed > self.last_finalized,
            "season_done": self.last_finalized >= 38,
        }


def summarize(rows: Iterable[Mapping], season: str) -> Gameweeks:
    """Require every fixture to be finished AND checked before finalization.

    Missing season or review evidence never finalizes a fixture. Adapters for
    known historical formats must supply those facts explicitly.
    """
    events: dict[int, list[tuple[bool, bool]]] = {}
    for row in rows:
        if str(row.get("season", "")) != season:
            continue
        try:
            event = float(row.get("event", ""))
            if not event.is_integer() or not 1 <= event <= 38:
                continue
        except (TypeError, ValueError, OverflowError):
            continue
        finished = truthy(row.get("finished"))
        events.setdefault(int(event), []).append(
            (finished, finished and truthy(row.get("data_checked"))))
    return Gameweeks(
        tuple(sorted(gw for gw, flags in events.items() if all(f[0] for f in flags))),
        tuple(sorted(gw for gw, flags in events.items() if all(f[1] for f in flags))),
    )


def from_frame(fixtures, season: str) -> Gameweeks:
    """DataFrame adapter; season-less in-memory inputs belong to the caller's season."""
    rows = fixtures.to_dict("records")
    if "season" not in fixtures.columns:
        rows = [dict(row, season=season) for row in rows]
    return summarize(rows, season)
