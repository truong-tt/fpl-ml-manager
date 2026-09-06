from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import backtest
import chips
import data_loader
import gameweeks
import main
import model_preparation as models
import season_replay
import workflow_gate

SEASON = main.SEASON


class FinalizationParityTests(unittest.TestCase):
    def test_every_caller_uses_the_same_finalization_evidence(self):
        rows = [
            dict(season="2025-2026", event=38, finished=True, data_checked=True),
            dict(season=SEASON, event=1, finished="yes", data_checked="1"),
            dict(season=SEASON, event=2, finished=True, data_checked=True),
            dict(season=SEASON, event=2, finished=True, data_checked=False),
            dict(season=SEASON, event=3, finished=False, data_checked=False),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixtures.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            frame = pd.read_csv(path)
            status = workflow_gate.fixture_status(path, SEASON)
            self.assertEqual(status, dict(last_completed=2, last_finalized=1,
                                          review_pending=True, season_done=False))
            self.assertEqual(main._last_finished_gw(frame), 1)
            self.assertEqual(main._last_completed_gw(frame), 2)
            self.assertEqual(backtest._finished_gws(frame), [1])
            self.assertEqual(season_replay._last_finished_gw(frame, SEASON), 1)

    def test_missing_review_never_finalizes(self):
        frame = pd.DataFrame([dict(season=SEASON, event=38, finished=True)])
        self.assertFalse(main._season_complete(frame))
        self.assertEqual(backtest._finished_gws(frame), [])
        self.assertEqual(season_replay._last_finished_gw(frame, SEASON), 0)

    def test_invalid_events_and_missing_season_cannot_close_season(self):
        rows = [dict(season=SEASON, event=e, finished=True, data_checked=True)
                for e in ("nan", "inf", "3.5", "", 0, 39)]
        rows.append(dict(event=38, finished=True, data_checked=True))
        self.assertEqual(gameweeks.summarize(rows, SEASON).finalized, ())

    def test_gate_runs_without_site_packages(self):
        result = subprocess.run(
            [sys.executable, "-S", str(ROOT / "src/workflow_gate.py"), "season",
             "--fixtures", "does-not-exist.csv"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("season_done=false", result.stdout)

    def test_summary_requires_finished_and_reviewed(self):
        frame = pd.DataFrame([dict(id=1, finished="yes", data_checked="1"),
                              dict(id=2, finished=True, data_checked=False)])
        self.assertEqual(data_loader.finalized_gws(frame), [1])
        self.assertEqual(data_loader.finalized_gws(frame.drop(columns="finished")), [])

    def test_missing_finalized_history_fails_ingestion(self):
        with patch.object(data_loader, "_fetch_gw_csv", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "Finalized GW2 player history unavailable"):
                data_loader._build_history_current([2], pd.DataFrame(), {}, pd.DataFrame())


class ChipDecisionTests(unittest.TestCase):
    def test_revision_withdrawal_and_transfer_carry_through_one_call(self):
        proj = pd.DataFrame(dict(id=[1, 2], pos_id=[3, 4],
                                 cap_xp_8=[6., 5.], xp_8=[5., 4.]))
        fixtures = pd.DataFrame(dict(event=[8], team_h=[1], team_a=[2]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chips.json"
            chips.save_chip_state({"tc1": 8, "bb1": 2}, path)
            decision = chips.decide_chips(
                proj, fixtures, 8, {1, 2}, {1, 2}, horizon=1,
                transfers_in=[3, 4, 5, 6], hits=0, free_transfers=3, path=path)
            self.assertEqual(decision.played, "wc1")
            self.assertEqual(decision.next_ft, 4)
            self.assertEqual(chips.load_chip_state(path), {"bb1": 2, "wc1": 8})
            decision = chips.decide_chips(
                proj, fixtures, 8, {1, 2}, {1, 2}, horizon=1,
                transfers_in=[3], hits=0, free_transfers=3, path=path)
            self.assertIsNone(decision.played)
            self.assertEqual(decision.next_ft, 3)
            self.assertEqual(chips.load_chip_state(path), {"bb1": 2})


class ModelPreparationTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        for name, value in dict(DATA_DIR=self.root,
                                MODEL_STATE_PATH=self.root / "state.json",
                                POINTS_RECALIB_PATH=self.root / "points.json",
                                MINUTES_RECALIB_PATH=self.root / "minutes.json").items():
            self.stack.enter_context(patch.object(models, name, value))
        self.events = []
        self.frames = [pd.DataFrame([dict(season=SEASON, event=2, finished=True,
                                          data_checked=True)])] + [pd.DataFrame()] * 3
        self.families = {
            "match": ([f"xgb_{s}_goals.json" for s in ("home", "away")], models.match_feature_cols()),
            "points": ([f"xgb_points_q{q}_p{p}.json" for q in (10, 50, 90)
                        for p in (1, 2, 3, 4)], models._pos_feature_cols()),
            "minutes": (["xgb_minutes_plays.json", "xgb_minutes_when_played.json"], models.minutes_feature_cols()),
            "bonus": ([f"xgb_bonus_q{q}.json" for q in (10, 50, 90)], models.points_feature_cols()),
        }
        for kind, name in (("match", "train_match_models"), ("points", "train_points_models"),
                           ("minutes", "train_minutes_model"), ("bonus", "train_bonus_model")):
            self.stack.enter_context(patch.object(models, name,
                side_effect=lambda *args, k=kind: self.train(k)))
        self.stack.enter_context(patch.object(models, "compute_fixture_lambdas",
                                              side_effect=lambda *args: self.events.append("lambdas")))

        def calibrate(*args):
            self.events.append("calibration")
            models.POINTS_RECALIB_PATH.write_text("{}")
            models.MINUTES_RECALIB_PATH.write_text("{}")

        def load(*args):
            self.assertTrue(models.POINTS_RECALIB_PATH.exists())
            self.assertTrue(models.MINUTES_RECALIB_PATH.exists())
            self.events.append("load")
            return SimpleNamespace(points_models={}, minutes_model={}, bonus_models={})

        self.stack.enter_context(patch.object(models, "_maybe_recalibrate", side_effect=calibrate))
        self.stack.enter_context(patch.object(models, "FPLEngine", side_effect=load))

    def train(self, kind):
        self.events.append(kind)
        files, cols = self.families[kind]
        frame = pd.DataFrame([[0.] * len(cols), [1.] * len(cols)], columns=cols)
        booster = xgb.train({"objective": "reg:squarederror", "nthread": 1},
                            xgb.DMatrix(frame, label=[0., 1.]), num_boost_round=1)
        for name in files:
            booster.save_model(self.root / name)

    def test_preparation_orders_dependencies_then_marks_readiness(self):
        models.prepare_engine(*self.frames)
        self.assertEqual(self.events, ["match", "lambdas", "points", "minutes", "bonus",
                                       "calibration", "load"])
        self.assertEqual(json.loads(models.MODEL_STATE_PATH.read_text()),
                         dict(season=SEASON, finalized_through_gw=2))
        self.events.clear()
        models.prepare_engine(*self.frames)
        self.assertEqual(self.events, ["lambdas", "calibration", "load"])

    def test_corruption_in_non_probe_artifact_triggers_repair(self):
        models.prepare_engine(*self.frames)
        (self.root / "xgb_points_q90_p4.json").write_text("broken")
        self.events.clear()
        models.prepare_engine(*self.frames)
        self.assertIn("points", self.events)

    def test_silent_trainer_cannot_mark_missing_models_ready(self):
        with patch.object(models, "train_points_models"):
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                models.prepare_engine(*self.frames)
        self.assertFalse(models.MODEL_STATE_PATH.exists())

    def test_calibration_failure_does_not_advance_marker(self):
        with patch.object(models, "_maybe_recalibrate", side_effect=RuntimeError("calibration failed")):
            with self.assertRaisesRegex(RuntimeError, "calibration failed"):
                models.prepare_engine(*self.frames)
        self.assertFalse(models.MODEL_STATE_PATH.exists())

    def test_schema_order_is_part_of_readiness(self):
        self.train("match")
        path = self.root / "xgb_home_goals.json"
        expected = self.families["match"][1]
        self.assertTrue(models._schema_drift(path, list(reversed(expected))))

    def test_partial_repair_cannot_leave_the_old_readiness_marker(self):
        models.prepare_engine(*self.frames)
        (self.root / "xgb_points_q10_p1.json").write_text("broken")

        def fail_after_writing():
            self.train("points")
            raise RuntimeError("training interrupted")

        with patch.object(models, "train_points_models", side_effect=fail_after_writing):
            with self.assertRaisesRegex(RuntimeError, "training interrupted"):
                models.prepare_engine(*self.frames)
        self.assertFalse(models.MODEL_STATE_PATH.exists())
        self.events.clear()
        models.prepare_engine(*self.frames)
        self.assertIn("points", self.events)

    def test_real_trainers_reject_empty_training_instead_of_reusing_stale_heads(self):
        import train_points_model
        import train_bonus_model
        import train_minutes_model
        for module, function in ((train_points_model, "train_points_models"),
                                 (train_bonus_model, "train_bonus_model"),
                                 (train_minutes_model, "train_minutes_model")):
            with self.subTest(function=function), ExitStack() as stack:
                stack.enter_context(patch.object(module.pd, "read_csv", return_value=pd.DataFrame()))
                stack.enter_context(patch.object(module, "build_match_features", return_value=pd.DataFrame()))
                stack.enter_context(patch.object(module, "build_player_features",
                    return_value=pd.DataFrame(columns=["target"])))
                with self.assertRaisesRegex(RuntimeError, "Insufficient"):
                    getattr(module, function)()


class PipelineOutcomeTests(unittest.TestCase):
    def test_in_play_run_cannot_advertise_a_fresh_lineup(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            pd.DataFrame([dict(season=SEASON, event=3, finished=False,
                               data_checked=False, kickoff_time=pd.Timestamp.now(tz="UTC"))]).to_csv(
                                   root / "fixtures.csv", index=False)
            for name in ("history", "players", "teams"):
                pd.DataFrame({"id": [1]}).to_csv(root / f"{name}.csv", index=False)
            (root / "lineup.md").write_text("previous lineup")
            stack.enter_context(patch.object(main, "DATA_DIR", root))
            stack.enter_context(patch.object(main, "OUT_DIR", root))
            stack.enter_context(patch.object(main, "refresh_data"))
            prepare = stack.enter_context(patch.object(main, "prepare_engine"))
            stack.enter_context(patch.dict(os.environ, GITHUB_OUTPUT=str(root / "output"),
                                           GITHUB_STEP_SUMMARY=str(root / "summary")))
            main.main()
            prepare.assert_not_called()
            self.assertIn("generated=false", (root / "output").read_text())
            self.assertIn("reason=in_play", (root / "output").read_text())
            self.assertEqual((root / "lineup.md").read_text(), "previous lineup")


if __name__ == "__main__":
    unittest.main()
