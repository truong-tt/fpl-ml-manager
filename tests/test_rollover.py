from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import data_loader
import workflow_gate


class FinalizationTests(unittest.TestCase):
    def test_fixture_gate_separates_completed_from_finalized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixtures.csv"
            with path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh, fieldnames=["season", "event", "finished", "data_checked"]
                )
                writer.writeheader()
                writer.writerows([
                    {"season": "2025-2026", "event": 38, "finished": True,
                     "data_checked": True},
                    {"season": "2026-2027", "event": 1, "finished": True,
                     "data_checked": False},
                ])
            status = workflow_gate.fixture_status(path, "2026-2027")
            self.assertEqual(status["last_completed"], 1)
            self.assertEqual(status["last_finalized"], 0)
            self.assertTrue(status["review_pending"])
            self.assertFalse(status["season_done"])

    def test_summary_requires_finished_and_checked(self) -> None:
        summary = pd.DataFrame([
            {"id": 1, "finished": True, "data_checked": True},
            {"id": 2, "finished": True, "data_checked": False},
            {"id": 3, "finished": False, "data_checked": True},
        ])
        self.assertEqual(data_loader.finalized_gws(summary), [1])

    def test_legacy_replay_state_does_not_block_new_season(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "replay.csv"
            path.write_text("gw,gw_total\n38,60\n", encoding="utf-8")
            self.assertEqual(workflow_gate.last_replayed(path, "2026-2027"), 0)

    def test_replay_gate_advances_once_when_scores_become_checked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixtures = root / "fixtures.csv"
            replay = root / "replay.csv"
            replay.write_text("season,gw\n2026-2027,1\n", encoding="utf-8")

            def write_fixture(checked: bool) -> None:
                fixtures.write_text(
                    "season,event,finished,data_checked\n"
                    f"2026-2027,2,True,{checked}\n",
                    encoding="utf-8",
                )

            write_fixture(False)
            status = workflow_gate.fixture_status(fixtures, "2026-2027")
            self.assertEqual(status["last_finalized"], 0)
            self.assertFalse(status["last_finalized"] >
                             workflow_gate.last_replayed(replay, "2026-2027"))

            write_fixture(True)
            status = workflow_gate.fixture_status(fixtures, "2026-2027")
            self.assertEqual(status["last_finalized"], 2)
            self.assertTrue(status["last_finalized"] >
                            workflow_gate.last_replayed(replay, "2026-2027"))

            replay.write_text("season,gw\n2026-2027,1\n2026-2027,2\n",
                              encoding="utf-8")
            self.assertFalse(status["last_finalized"] >
                             workflow_gate.last_replayed(replay, "2026-2027"))


class SeasonStateTests(unittest.TestCase):
    def test_walk_forward_split_is_season_aware(self) -> None:
        try:
            import backtest
        except ModuleNotFoundError as exc:
            self.skipTest(f"optional ML dependency unavailable: {exc.name}")
        rows = pd.DataFrame([
            {"season": "2025-2026", "round": 38, "value": "prior"},
            {"season": "2026-2027", "round": 1, "value": "test"},
            {"season": "2026-2027", "round": 2, "value": "future"},
        ])
        train, test = backtest._season_split(rows, "2026-2027", 1, "round")
        self.assertEqual(train["value"].tolist(), ["prior"])
        self.assertEqual(test["value"].tolist(), ["test"])

    def test_model_marker_refreshes_once_per_finalized_gw(self) -> None:
        try:
            import main
        except ModuleNotFoundError as exc:
            self.skipTest(f"optional ML dependency unavailable: {exc.name}")
        current = {"season": main.SEASON, "finalized_through_gw": 4}
        self.assertFalse(main._models_need_refresh(current, 4))
        self.assertTrue(main._models_need_refresh(current, 5))
        self.assertTrue(main._models_need_refresh({"season": "2025-2026"}, 0))

    def test_legacy_snapshot_is_ignored(self) -> None:
        try:
            import main
        except ModuleNotFoundError as exc:
            self.skipTest(f"optional ML dependency unavailable: {exc.name}")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            pd.DataFrame([{"id": 1, "bank": 1.0, "free_transfers": 2}]).to_csv(
                out / "squad_snapshot.csv", index=False
            )
            with patch.object(main, "OUT_DIR", out):
                self.assertIsNone(main._load_prior())

    def test_matching_snapshot_resumes_and_mismatched_snapshot_resets(self) -> None:
        try:
            import main
        except ModuleNotFoundError as exc:
            self.skipTest(f"optional ML dependency unavailable: {exc.name}")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            snap = out / "squad_snapshot.csv"
            pd.DataFrame([{
                "season": main.SEASON, "id": 11, "bank": 2.5,
                "free_transfers": 3,
            }]).to_csv(snap, index=False)
            with patch.object(main, "OUT_DIR", out):
                self.assertEqual(main._load_prior(), ({11}, 2.5, 3))

            pd.DataFrame([{
                "season": "2025-2026", "id": 11, "bank": 2.5,
                "free_transfers": 3,
            }]).to_csv(snap, index=False)
            with patch.object(main, "OUT_DIR", out):
                self.assertIsNone(main._load_prior())


class ArchiveLoaderTests(unittest.TestCase):
    def test_per_gw_archive_remaps_stable_player_and_team_codes(self) -> None:
        base = pd.DataFrame([{
            "player_id": 10, "player_code": 1000, "team_code": 50,
        }])
        summary = pd.DataFrame([{"id": 1}])
        gw = pd.DataFrame([{
            "id": 10, "minutes": 90, "total_points": 8, "bonus": 2,
            "bps": 30, "assists": 1, "defensive_contribution": 0,
        }])
        gw_players = pd.DataFrame([{
            "player_id": 10, "player_code": 1000, "team_code": 60,
        }])
        players = pd.DataFrame([{"id": 99, "code": 1000}])
        teams = pd.DataFrame([
            {"team_id": 7, "code": 50}, {"team_id": 8, "code": 60},
        ])
        fixtures = pd.DataFrame([{"id": 3, "team_h": 8, "team_a": 7}])

        def fetch_csv(path: str, cache: bool = True):
            if path.endswith("players.csv"):
                return base
            if path.endswith("gameweek_summaries.csv"):
                return summary
            return None

        def fetch_gw(_gw: int, filename: str, cache_history: bool,
                     season: str = data_loader.SEASON):
            return gw_players if filename == "players.csv" else gw

        with patch.object(data_loader, "_fetch_csv", side_effect=fetch_csv), \
             patch.object(data_loader, "_fetch_gw_csv", side_effect=fetch_gw), \
             patch.object(data_loader, "_build_opta_per_gw",
                          return_value=pd.DataFrame()):
            result = data_loader._build_history_per_gw_archive(
                "2025-2026", players, teams, {(8, 1, "2025-2026"): 3}, fixtures
            )

        self.assertEqual(len(result), 1)
        self.assertEqual(int(result.iloc[0]["player_id"]), 99)
        self.assertEqual(int(result.iloc[0]["team"]), 8)
        self.assertEqual(float(result.iloc[0]["total_points"]), 8)
        self.assertEqual(float(result.iloc[0]["bonus"]), 2)


if __name__ == "__main__":
    unittest.main()
