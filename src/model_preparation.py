"""Prepare validated model artifacts and calibration before loading projections."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import xgboost as xgb

from data_loader import SEASON
from features import match_feature_cols, minutes_feature_cols, points_feature_cols
from fpl_engine import FPLEngine
from gameweeks import from_frame
from train_bonus_model import train_bonus_model
from train_match_model import compute_fixture_lambdas, train_match_models
from train_minutes_model import train_minutes_model
from train_points_model import _pos_feature_cols, train_points_models

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# Auto-recalibration cadence. Pipeline runs 2×/day on data-source refresh, but
# walk-forward retrain is expensive (~minutes) and recalib drift is slow. Re-fit
# only when JSON older than N days OR missing. Manual override: delete the JSON.
RECALIB_STALE_DAYS = 14
RECALIB_HOLDOUT_K = 8
POINTS_RECALIB_PATH = DATA_DIR / "points_recalib.json"
MINUTES_RECALIB_PATH = DATA_DIR / "minutes_recalib.json"
MODEL_STATE_PATH = DATA_DIR / "model_state.json"


def _booster_features(path: Path) -> list[str] | None:
    """Booster's stored feature_names. None if file missing or unreadable."""
    if not path.exists():
        return None
    try:
        b = xgb.Booster()
        b.load_model(path)
        return list(b.feature_names or [])
    except Exception:
        return None


def _schema_drift(path: Path, expected: list[str]) -> bool:
    """Missing, unreadable or differently ordered features require retraining."""
    feats = _booster_features(path)
    if feats is None:
        return True
    return feats != list(expected)


def _invalidate_recalib(*paths: Path) -> None:
    """Drop stale recalib JSONs after retrain. New raw quantiles → old map invalid."""
    for p in paths:
        if p.exists():
            p.unlink()


def _models_need_refresh(state: dict, finalized_gw: int) -> bool:
    try:
        trained_gw = int(state.get("finalized_through_gw", -1))
    except (AttributeError, TypeError, ValueError):
        return True
    return state.get("season") != SEASON or trained_gw != finalized_gw


def _ensure_models(fx: pd.DataFrame, hist: pd.DataFrame, teams: pd.DataFrame) -> None:
    """Train missing match / points / minutes / bonus artifacts.

    Match → fixture λ → points + bonus. λ feeds points feature schema, so
    refresh fixture_lambdas.csv any time match models present (cheap). Also
    detect feature-schema drift on cached on-disk boosters (e.g. features.py
    grew new cols since last train) and force retrain.
    """
    finalized_gw = from_frame(fx, SEASON).last_finalized
    try:
        state = json.loads(MODEL_STATE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        state = {}
    refresh = _models_need_refresh(state, finalized_gw)

    match_files = ("xgb_home_goals.json", "xgb_away_goals.json")
    match_have = all((DATA_DIR / f).exists() for f in match_files)
    if (refresh or not match_have
            or any(_schema_drift(DATA_DIR / f, match_feature_cols()) for f in match_files)):
        print("[models] (re)training match head — schema drift or missing artifacts")
        MODEL_STATE_PATH.unlink(missing_ok=True)
        train_match_models(fx, hist, teams)
    compute_fixture_lambdas(fx, hist, teams)
    points_files = [f"xgb_points_q{q:02d}_p{p}.json"
                    for q in (10, 50, 90) for p in (1, 2, 3, 4)]
    if (refresh or not all((DATA_DIR / f).exists() for f in points_files)
            or any(_schema_drift(DATA_DIR / f, _pos_feature_cols()) for f in points_files)):
        print("[models] (re)training points head — schema drift or missing artifacts")
        MODEL_STATE_PATH.unlink(missing_ok=True)
        _invalidate_recalib(POINTS_RECALIB_PATH)
        train_points_models()
    minutes_files = ("xgb_minutes_plays.json", "xgb_minutes_when_played.json")
    minutes_have_two_stage = all((DATA_DIR / f).exists() for f in minutes_files)
    if (refresh or not minutes_have_two_stage
            or any(_schema_drift(DATA_DIR / f, minutes_feature_cols()) for f in minutes_files)):
        print("[models] (re)training minutes head — schema drift or missing artifacts")
        MODEL_STATE_PATH.unlink(missing_ok=True)
        _invalidate_recalib(MINUTES_RECALIB_PATH)
        train_minutes_model()
    bonus_files = [f"xgb_bonus_q{q:02d}.json" for q in (10, 50, 90)]
    if (refresh or not all((DATA_DIR / f).exists() for f in bonus_files)
            or any(_schema_drift(DATA_DIR / f, points_feature_cols()) for f in bonus_files)):
        print("[models] (re)training bonus head — schema drift or missing artifacts")
        MODEL_STATE_PATH.unlink(missing_ok=True)
        train_bonus_model()
    # Trainers may return without producing files when training data is empty.
    # Never mark that attempt ready or silently fall back to missing heads.
    for files, expected in ((match_files, match_feature_cols()),
                            (points_files, _pos_feature_cols()),
                            (minutes_files, minutes_feature_cols()),
                            (bonus_files, points_feature_cols())):
        invalid = [f for f in files if _schema_drift(DATA_DIR / f, expected)]
        if invalid:
            raise RuntimeError(f"Model preparation incomplete: {invalid}")


def _recalib_stale(path: Path, max_age_days: int = RECALIB_STALE_DAYS) -> bool:
    """True if recalib JSON missing or older than max_age_days."""
    if not path.exists():
        return True
    age_days = (time.time() - path.stat().st_mtime) / 86400.0
    return age_days > max_age_days


def _maybe_recalibrate(fixtures: pd.DataFrame, history: pd.DataFrame,
                       players: pd.DataFrame, teams: pd.DataFrame) -> None:
    """Re-fit points + minutes recalib JSONs if stale.

    Walk-forward retrain runs on last RECALIB_HOLDOUT_K finished GWs. Outputs
    consumed by predict_quantiles / predict_minutes auto-load on next inference
    pass within same process — but they cache via load_*_models, so recalib is
    re-read at next model-load. Engine constructed AFTER this call.
    """
    points_stale = _recalib_stale(POINTS_RECALIB_PATH)
    minutes_stale = _recalib_stale(MINUTES_RECALIB_PATH)
    if not (points_stale or minutes_stale):
        return

    # Lazy imports. Backtest pulls xgb + heavy training stack.
    from backtest import (_resolve_holdout, walk_forward_minutes,
                          walk_forward_points)
    from recalibrate_minutes import fit_minutes_recalib
    from recalibrate_minutes import save_recalib as save_min_recalib
    from recalibrate_points import fit_points_recalib
    from recalibrate_points import save_recalib as save_pts_recalib

    try:
        holdout = _resolve_holdout(fixtures, RECALIB_HOLDOUT_K, None, None)
    except RuntimeError:
        print("recalib skipped: no finished GWs available")
        return
    if not holdout:
        print("recalib skipped: empty holdout")
        return
    print(f"recalib: walk-forward over GWs {holdout}")

    if points_stale:
        pred = walk_forward_points(holdout, history, fixtures, players, teams)
        if not pred.empty:
            coef = fit_points_recalib(pred, played_only=True)
            save_pts_recalib(coef, POINTS_RECALIB_PATH)
            print(f"recalib points -> {POINTS_RECALIB_PATH}")

    if minutes_stale:
        pred = walk_forward_minutes(holdout, history, fixtures, players, teams)
        if not pred.empty:
            coef = fit_minutes_recalib(pred)
            save_min_recalib(coef, MINUTES_RECALIB_PATH)
            print(f"recalib minutes -> {MINUTES_RECALIB_PATH}")


def prepare_engine(fixtures: pd.DataFrame, history: pd.DataFrame,
                   players: pd.DataFrame, teams: pd.DataFrame) -> FPLEngine:
    """Return a projection engine only after model and calibration readiness.

    A failed preparation never advances the finalized-Gameweek marker, so the
    next attempt retries instead of treating partial work as a completed refresh.
    """
    _ensure_models(fixtures, history, teams)
    _maybe_recalibrate(fixtures, history, players, teams)
    engine = FPLEngine(fixtures, history, players, teams)
    if any(head is None for head in
           (engine.points_models, engine.minutes_model, engine.bonus_models)):
        raise RuntimeError("Model preparation failed to load all prediction heads")
    MODEL_STATE_PATH.write_text(json.dumps({
        "season": SEASON,
        "finalized_through_gw": from_frame(fixtures, SEASON).last_finalized,
    }, indent=2) + "\n", encoding="utf-8")
    return engine
