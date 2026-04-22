"""FPL bootstrap + fixture + per-player history fetcher."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://fantasy.premierleague.com/api/"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

HIST_NUM = [
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "bps", "ict_index", "saves",
    "clearances_blocks_interceptions", "tackles", "recoveries",
    "minutes", "goals_scored", "goals_conceded", "total_points",
]


def _get(endpoint: str, retries: int = 3) -> dict | None:
    """Retries with linear backoff; returns None on repeated failure."""
    for i in range(retries):
        try:
            r = requests.get(f"{BASE}{endpoint}", timeout=15)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            time.sleep(1.5 * (i + 1))
    return None


def _num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Coerces listed cols to float, fills NaN with 0."""
    for c in cols:
        df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0.0)
    return df


def main() -> None:
    """Refreshes players / teams / fixtures / history CSVs under data/."""
    boot = _get("bootstrap-static/")
    if not boot:
        raise RuntimeError("bootstrap-static fetch failed")

    teams = pd.DataFrame(boot["teams"])[[
        "id", "name", "short_name", "strength",
        "strength_overall_home", "strength_overall_away",
    ]].rename(columns={"id": "team_id", "name": "team_name"})

    players = pd.DataFrame(boot["elements"])
    players = _num(players, [
        "selected_by_percent", "transfers_in_event", "transfers_out_event",
        "penalties_order", "direct_freekicks_order",
        "corners_and_indirect_freekicks_order",
    ])

    fx = pd.DataFrame(_get("fixtures/") or [])
    keep = ["id", "event", "finished", "kickoff_time", "team_h", "team_a",
            "team_h_score", "team_a_score"]
    fx = fx[[c for c in keep if c in fx.columns]].dropna(subset=["event"])
    fx["event"] = fx["event"].astype(int)

    active = players[(players["minutes"] > 0) | (players["now_cost"] > 40)]["id"].tolist()
    rows: list[pd.DataFrame] = []
    for pid in active:
        h = _get(f"element-summary/{pid}/")
        if h and "history" in h:
            d = pd.DataFrame(h["history"])
            d["player_id"] = pid
            rows.append(d)
    history = _num(pd.concat(rows, ignore_index=True), HIST_NUM) if rows else pd.DataFrame()

    players.to_csv(DATA_DIR / "players.csv", index=False)
    teams.to_csv(DATA_DIR / "teams.csv", index=False)
    fx.to_csv(DATA_DIR / "fixtures.csv", index=False)
    history.to_csv(DATA_DIR / "history.csv", index=False)


if __name__ == "__main__":
    main()