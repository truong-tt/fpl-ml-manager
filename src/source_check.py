"""Live source smoke check using the production current-season normalization path.

No model training or squad writes. A temporary cache prevents stale local data
from making an unavailable source look healthy.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import requests

import data_loader as loader
from gameweeks import from_frame


def check_sources() -> dict:
    original_cache = loader.CACHE_DIR
    with tempfile.TemporaryDirectory(prefix="fpl-source-check-") as tmp:
        loader.CACHE_DIR = Path(tmp)
        try:
            current, maximum, summary = loader._discover_gw_bounds()
            if summary.empty:
                raise RuntimeError(f"No official Gameweek summary for {loader.SEASON}")
            print(f"[sources] {loader.SEASON}: current GW{current}, checking normalization", flush=True)
            teams = loader._build_teams()
            codes = dict(zip(teams["code"].astype(int), teams["team_id"].astype(int)))
            players = loader._build_players(teams, current)
            fixtures = loader._build_fixtures_current(current, maximum, codes, summary)
            fixtures, lookup = loader._assign_global_fixture_ids(fixtures)
            checked = loader.finalized_gws(summary)
            history = loader._build_history_current(checked, players, lookup, fixtures)
            if len(teams) != 20 or players.empty or fixtures.empty:
                raise RuntimeError("Current-season teams, players or fixtures are incomplete")
            if set(checked) - set(history["round"].unique()):
                raise RuntimeError("Finalized Gameweeks are missing player history")
            if not history.empty and (history["fixture"] == 0).any():
                raise RuntimeError("Current player history has unmapped fixtures")
            if not checked and not history.empty:
                raise RuntimeError("Unfinalized player history entered the training pool")
            finalized = list(from_frame(fixtures, loader.SEASON).finalized)
            if finalized != checked:
                raise RuntimeError("Fixture finalization disagrees with the official summary")
            result = {
                "season": loader.SEASON, "current_gw": current,
                "teams": len(teams), "players": len(players),
                "fixtures": len(fixtures), "history_rows": len(history),
                "finalized_gws": finalized,
                "source": f"{loader.FPL_CI_BASE}/{loader.SEASON}",
            }
            # This endpoint is an optional overlay in production. Expose failure
            # explicitly without pretending the primary ingest depends on it.
            try:
                response = requests.get(loader.FPL_API_BASE + "bootstrap-static/", timeout=30)
                response.raise_for_status()
                live = response.json()
                if not live.get("elements") or not live.get("events"):
                    raise ValueError("empty live player or Gameweek data")
                years = {str(e.get("deadline_time", ""))[:4] for e in live["events"]}
                if not set(loader.SEASON.split("-")).issubset(years):
                    raise ValueError("live Gameweek dates do not match the requested season")
                response = requests.get(loader.FPL_API_BASE + "fixtures/", timeout=30)
                response.raise_for_status()
                live_fixtures = response.json()
                if not live_fixtures:
                    raise ValueError("empty live fixture data")
                result["live_overlay"] = "available"
                result["live_players"] = len(live["elements"])
                result["live_fixtures"] = len(live_fixtures)
            except (requests.RequestException, ValueError, KeyError) as exc:
                result["live_overlay"] = f"unavailable: {type(exc).__name__}: {exc}"
            return result
        finally:
            loader.CACHE_DIR = original_cache


def main() -> None:
    result = check_sources()
    report = json.dumps(result, indent=2)
    print(report)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(f"### Current-season source check\n\n```json\n{report}\n```\n")


if __name__ == "__main__":
    main()
