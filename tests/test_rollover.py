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

import chips
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
    def test_preseason_projection_uses_latest_prior_season_baseline(self) -> None:
        import fpl_engine

        engine = fpl_engine.FPLEngine.__new__(fpl_engine.FPLEngine)
        engine.fixtures = engine.history = engine.players = engine.teams = pd.DataFrame()
        past = pd.DataFrame([
            {"player_id": 1, "season": "2024-2025", "round": 38, "value": 1},
            {"player_id": 1, "season": "2025-2026", "round": 38, "value": 2},
        ])
        with patch.object(fpl_engine, "build_match_features",
                          return_value=pd.DataFrame()), \
             patch.object(fpl_engine, "build_player_features", return_value=past):
            latest = engine._latest_rolling()
        self.assertEqual(int(latest.loc[1, "value"]), 2)

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


class ChipRuleTests(unittest.TestCase):
    """FPL 2026/27: two chip sets, set 1 expires at the GW19 deadline."""

    def test_chip_set_boundaries(self) -> None:
        self.assertEqual(chips.chip_set(1), 1)
        self.assertEqual(chips.chip_set(19), 1)
        self.assertEqual(chips.chip_set(20), 2)
        self.assertEqual(chips.chip_set(38), 2)
        self.assertEqual(chips.set_last_gw(5), 19)
        self.assertEqual(chips.set_last_gw(25), 38)

    def test_used_set1_chip_does_not_block_set2(self) -> None:
        used = {"tc1": 5}
        self.assertFalse(chips.chip_available("tc", 12, used))
        self.assertTrue(chips.chip_available("tc", 25, used))

    def test_missing_chip_state_file_means_nothing_used(self) -> None:
        self.assertEqual(chips.load_chip_state(Path("does-not-exist.json")), {})

    def test_recommenders_never_cross_the_set_boundary(self) -> None:
        """From GW16 with H=8 the scan reaches GW23 — set-1 chips must not
        be proposed for GW20+, where set 1 has already expired."""
        proj = pd.DataFrame({
            "id": [1, 2],
            "pos_id": [3, 4],
            **{f"cap_xp_{t}": [1.0, 50.0 if t >= 20 else 2.0] for t in range(16, 24)},
            **{f"xp_{t}": [1.0, 50.0 if t >= 20 else 2.0] for t in range(16, 24)},
        })
        tc = chips.recommend_triple_captain(proj, {1, 2}, 16)
        self.assertIsNotNone(tc["gw"])
        self.assertLessEqual(tc["gw"], 19)

        bb = chips.recommend_bench_boost(proj, {1, 2}, set(), 16)
        self.assertIsNotNone(bb["gw"])
        self.assertLessEqual(bb["gw"], 19)

    def test_spent_chip_yields_no_recommendation(self) -> None:
        proj = pd.DataFrame({"id": [1], "pos_id": [3],
                             "cap_xp_5": [9.0], "xp_5": [9.0]})
        self.assertIsNone(
            chips.recommend_triple_captain(proj, {1}, 5, {"tc1": 2})["gw"])
        self.assertIsNone(
            chips.recommend_bench_boost(proj, {1}, set(), 5, {"bb1": 2})["gw"])
        self.assertFalse(
            chips.recommend_wildcard([1, 2, 3, 4], 0, 5, {"wc1": 2})["recommend"])
        self.assertTrue(
            chips.recommend_wildcard([1, 2, 3, 4], 0, 25, {"wc1": 2})["recommend"])

    def test_free_hit_cannot_follow_a_free_hit(self) -> None:
        """FH1 at GW19, so FH2 (available from GW20) must skip GW20 itself."""
        fx = pd.DataFrame({
            "event": [20, 21, 21],
            "team_h": [1, 1, 3],
            "team_a": [2, 2, 4],
        })
        # GW20 blanks teams 3+4; GW21 blanks none. Without the rule GW20 wins.
        out = chips.recommend_free_hit(fx, 20, 4, {"fh1": 19})
        self.assertNotEqual(out["gw"], 20)

    def test_spent_free_hit_yields_no_recommendation(self) -> None:
        fx = pd.DataFrame({"event": [10, 10], "team_h": [1, 3], "team_a": [2, 4]})
        self.assertIsNone(chips.recommend_free_hit(fx, 10, 4, {"fh1": 10})["gw"])


class FreeTransferCarryTests(unittest.TestCase):
    """2026/27: WC/FH preserve banked FTs; transfers consume only what's spent."""

    @staticmethod
    def _carry(ft: int, n_in: int, gw: int, used: dict) -> int:
        import main
        return main.carry_free_transfers(ft, n_in, gw, used)

    def test_partial_spend_keeps_the_remainder(self) -> None:
        # 3 banked, spend 1 -> 2 left, +1 for the new GW = 3.
        self.assertEqual(self._carry(3, 1, 8, {}), 3)

    def test_full_spend_drops_to_one(self) -> None:
        self.assertEqual(self._carry(3, 3, 8, {}), 1)

    def test_wildcard_week_preserves_the_bank(self) -> None:
        self.assertEqual(self._carry(3, 12, 8, {"wc1": 8}), 4)

    def test_carry_caps_at_five(self) -> None:
        self.assertEqual(self._carry(5, 0, 8, {}), 5)


if __name__ == "__main__":
    unittest.main()
