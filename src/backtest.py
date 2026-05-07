"""Walk-forward CV harness for points and match models.

For each holdout GW G in [start_gw, end_gw], retrain both models using only rows
strictly before G. Predict on rows at G. Accumulate predictions for offline
calibration audit in `calibration.py`. Write CSVs + markdown report under
`data/processed/backtest/`.

Rolling features in features.py shift-1 partitioned. Build feature frame once
over full history = leakage-free, as long as target-bearing rows at round >= G
excluded from training. Split by `round` after feature construction rather than
rebuild features per holdout.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from features import (build_match_features, build_player_features,
                      match_feature_cols, minutes_feature_cols,
                      points_feature_cols)
from train_match_model import score_matrix
from train_points_model import POSITIONS, QUANTILES, _pos_feature_cols, _row_pos

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = DATA_DIR / "processed" / "backtest"

POINTS_PARAMS = dict(objective="reg:quantileerror", learning_rate=0.03, max_depth=3,
                     subsample=0.8, colsample_bytree=0.8, min_child_weight=30,
                     reg_alpha=0.5, reg_lambda=2.0, verbosity=0)
POINTS_ROUNDS = 400
MATCH_PARAMS = dict(objective="count:poisson", learning_rate=0.05, max_depth=4,
                    subsample=0.85, colsample_bytree=0.85, verbosity=0)
MATCH_ROUNDS = 200
MINUTES_PARAMS = dict(objective="reg:logistic", learning_rate=0.05, max_depth=4,
                      subsample=0.85, colsample_bytree=0.85, min_child_weight=20,
                      reg_alpha=0.3, reg_lambda=1.5, verbosity=0)
MINUTES_ROUNDS = 300


def _finished_gws(fixtures: pd.DataFrame) -> list[int]:
    fin = fixtures["finished"].astype(str).str.lower().isin(["true", "1"])
    g = fixtures.assign(_fin=fin).groupby("event")["_fin"].agg(["sum", "size"])
    full = g[g["sum"] == g["size"]].index.astype(int).tolist()
    return sorted(full)


def _resolve_holdout(fixtures: pd.DataFrame, k: int,
                     start: int | None, end: int | None) -> list[int]:
    finished = _finished_gws(fixtures)
    if not finished:
        raise RuntimeError("no fully finished GWs available for backtest")
    if start is not None and end is not None:
        return [g for g in finished if start <= g <= end]
    return finished[-k:]


def _train_points_for_holdout(train: pd.DataFrame, feat_cols: list[str]
                              ) -> dict[int, dict[float, xgb.Booster]]:
    """Train per-position × per-quantile boosters on train rows only."""
    models: dict[int, dict[float, xgb.Booster]] = {}
    train = train.copy()
    train["_pos"] = _row_pos(train)
    for pos in POSITIONS:
        sub = train[train["_pos"] == pos]
        if len(sub) < 200:
            continue
        X = sub[feat_cols].astype(float).fillna(0.0)
        y = sub["target"].astype(float)
        models[pos] = {}
        for q in QUANTILES:
            params = {**POINTS_PARAMS, "quantile_alpha": q}
            m = xgb.train(params, xgb.DMatrix(X, label=y), num_boost_round=POINTS_ROUNDS)
            models[pos][q] = m
    return models


def _predict_points(models: dict[int, dict[float, xgb.Booster]],
                    test: pd.DataFrame, feat_cols: list[str],
                    recalib: dict | None = None) -> pd.DataFrame:
    """Predict q10/q50/q90 with row-sort enforcement. Mirror train_points_model.

    If `recalib` provided, apply per-(pos, alpha) affine map post-sort,
    re-enforce non-crossing.
    """
    out = pd.DataFrame(0.0, index=test.index, columns=["q10_pred", "q50_pred", "q90_pred"])
    pos_series = _row_pos(test)
    Xf = test[feat_cols].astype(float).fillna(0.0)
    qmap = {0.10: "q10_pred", 0.50: "q50_pred", 0.90: "q90_pred"}
    for pos in POSITIONS:
        mask = (pos_series == pos)
        if not mask.any() or pos not in models:
            continue
        dm = xgb.DMatrix(Xf.loc[mask])
        for q, col in qmap.items():
            out.loc[mask, col] = models[pos][q].predict(dm)
    vals = out[["q10_pred", "q50_pred", "q90_pred"]].values.copy()
    vals.sort(axis=1)
    out[["q10_pred", "q50_pred", "q90_pred"]] = vals
    if recalib is not None:
        # Reuse public apply path. Rename columns to match q10/q50/q90 schema.
        from recalibrate_points import apply_recalib
        renamed = out.rename(columns={"q10_pred": "q10", "q50_pred": "q50", "q90_pred": "q90"})
        renamed = apply_recalib(recalib, pos_series.values, renamed)
        out = renamed.rename(columns={"q10": "q10_pred", "q50": "q50_pred", "q90": "q90_pred"})
    return out.clip(lower=-3.0, upper=25.0)


def walk_forward_points(holdout_gws: list[int], history: pd.DataFrame,
                        fixtures: pd.DataFrame, players: pd.DataFrame,
                        teams: pd.DataFrame,
                        recalib: dict | None = None) -> pd.DataFrame:
    """Per holdout GW, train on round < G. Predict on round == G.

    Return long frame: gw, player_id, pos_id, q10_pred, q50_pred, q90_pred, y.
    If `recalib` supplied, apply after raw quantile prediction.
    """
    fixture_feats = build_match_features(fixtures, history, teams)
    all_feats = build_player_features(history, players, fixture_feats).dropna(subset=["target"])
    if all_feats.empty:
        return pd.DataFrame()

    feat_cols = _pos_feature_cols()
    pos_series_all = _row_pos(all_feats)
    chunks: list[pd.DataFrame] = []
    for gw in holdout_gws:
        train = all_feats[all_feats["round"] < gw]
        test = all_feats[all_feats["round"] == gw]
        if train.empty or test.empty:
            continue
        models = _train_points_for_holdout(train, feat_cols)
        preds = _predict_points(models, test, feat_cols, recalib=recalib)
        rec = pd.DataFrame({
            "gw": gw,
            "player_id": test["player_id"].astype(int).values,
            "pos_id": pos_series_all.loc[test.index].astype(int).values,
            "y": test["target"].astype(float).values,
            "minutes": test["minutes"].astype(float).values if "minutes" in test.columns else 0.0,
        }, index=test.index)
        rec = pd.concat([rec, preds], axis=1)
        chunks.append(rec.reset_index(drop=True))
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def walk_forward_match(holdout_gws: list[int], fixtures: pd.DataFrame,
                       history: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Per holdout GW, train Poisson home/away on event < G. Predict event == G.

    Return: gw, fixture_id, team_h, team_a, lambda_h, lambda_a, gh, ga, cs_h_p,
    cs_a_p, cs_h_actual, cs_a_actual.
    """
    df = build_match_features(fixtures, history, teams)
    fin = df["finished"].astype(str).str.lower().isin(["true", "1"])
    past = df[fin].dropna(subset=["team_h_score", "team_a_score"])
    if past.empty:
        return pd.DataFrame()
    feat_cols = match_feature_cols()
    chunks: list[pd.DataFrame] = []
    for gw in holdout_gws:
        train = past[past["event"] < gw]
        test = past[past["event"] == gw]
        if train.empty or test.empty:
            continue
        Xtr = train[feat_cols].astype(float)
        Xte = test[feat_cols].astype(float)
        boosters: dict[str, xgb.Booster] = {}
        for label, side in (("team_h_score", "home"), ("team_a_score", "away")):
            m = xgb.train(MATCH_PARAMS,
                          xgb.DMatrix(Xtr, label=train[label].astype(int)),
                          num_boost_round=MATCH_ROUNDS)
            boosters[side] = m
        dm = xgb.DMatrix(Xte)
        lh = boosters["home"].predict(dm)
        la = boosters["away"].predict(dm)
        cs_h_p = np.zeros(len(test))
        cs_a_p = np.zeros(len(test))
        for i, (lh_i, la_i) in enumerate(zip(lh, la)):
            M = score_matrix(float(lh_i), float(la_i))
            cs_h_p[i] = M[:, 0].sum()
            cs_a_p[i] = M[0, :].sum()
        gh = test["team_h_score"].astype(int).values
        ga = test["team_a_score"].astype(int).values
        chunks.append(pd.DataFrame({
            "gw": gw,
            "fixture_id": test["id"].astype(int).values if "id" in test.columns else np.arange(len(test)),
            "team_h": test["team_h"].astype(int).values,
            "team_a": test["team_a"].astype(int).values,
            "lambda_h": lh,
            "lambda_a": la,
            "gh": gh,
            "ga": ga,
            "cs_h_p": cs_h_p,
            "cs_a_p": cs_a_p,
            "cs_h_actual": (ga == 0).astype(int),
            "cs_a_actual": (gh == 0).astype(int),
        }))
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def walk_forward_minutes(holdout_gws: list[int], history: pd.DataFrame,
                         fixtures: pd.DataFrame, players: pd.DataFrame,
                         teams: pd.DataFrame,
                         recalib: dict | None = None) -> pd.DataFrame:
    """Per holdout GW G, train minutes/90 regressor on round < G. Predict on G.

    Return long frame: gw, player_id, pos_id, mins_pred (∈[0,1]), mins_actual (raw),
    played (1 if mins_actual > 0). Training target = clip(minutes, 0, 90)/90.
    If `recalib` supplied, apply per-pos isotonic after raw prediction.
    """
    fixture_feats = build_match_features(fixtures, history, teams)
    all_feats = build_player_features(history, players, fixture_feats)
    if all_feats.empty:
        return pd.DataFrame()

    all_feats = all_feats.copy()
    all_feats["mins_target"] = (all_feats["minutes"].clip(upper=90) / 90.0).clip(0.0, 1.0)
    feat_cols = minutes_feature_cols()
    pos_series_all = _row_pos(all_feats)

    chunks: list[pd.DataFrame] = []
    for gw in holdout_gws:
        train = all_feats[all_feats["round"] < gw]
        test = all_feats[all_feats["round"] == gw]
        if train.empty or test.empty:
            continue
        Xtr = train[feat_cols].astype(float).fillna(0.0)
        Xte = test[feat_cols].astype(float).fillna(0.0)
        ytr = train["mins_target"].astype(float)
        m = xgb.train(MINUTES_PARAMS, xgb.DMatrix(Xtr, label=ytr),
                      num_boost_round=MINUTES_ROUNDS)
        pred = np.clip(m.predict(xgb.DMatrix(Xte)), 0.0, 1.0)
        pos_arr = pos_series_all.loc[test.index].astype(int).values
        if recalib is not None:
            from recalibrate_minutes import apply_recalib as _apply_min
            pred = _apply_min(recalib, pos_arr, pred)
        mins_actual = test["minutes"].astype(float).values
        chunks.append(pd.DataFrame({
            "gw": gw,
            "player_id": test["player_id"].astype(int).values,
            "pos_id": pos_arr,
            "mins_pred": pred,
            "mins_actual": mins_actual,
            "played": (mins_actual > 0).astype(int),
        }))
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward CV for points + match models")
    p.add_argument("--k", type=int, default=5,
                   help="Trailing finished GWs to hold out. Default 5")
    p.add_argument("--start", type=int, default=None, help="Holdout start GW. Inclusive")
    p.add_argument("--end", type=int, default=None, help="Holdout end GW. Inclusive")
    p.add_argument("--skip-points", action="store_true", help="Skip points walk-forward")
    p.add_argument("--skip-match", action="store_true", help="Skip match walk-forward")
    p.add_argument("--skip-minutes", action="store_true", help="Skip minutes walk-forward")
    p.add_argument("--recalib", type=Path, default=None,
                   help="Path to points_recalib.json. Apply after raw quantile prediction")
    p.add_argument("--minutes-recalib", type=Path, default=None,
                   help="Path to minutes_recalib.json. Apply per-pos isotonic to mins_pred")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fixtures = pd.read_csv(DATA_DIR / "fixtures.csv")
    history = pd.read_csv(DATA_DIR / "history.csv")
    players = pd.read_csv(DATA_DIR / "players.csv")
    teams = pd.read_csv(DATA_DIR / "teams.csv")
    holdout = _resolve_holdout(fixtures, args.k, args.start, args.end)
    if not holdout:
        raise RuntimeError("no holdout GWs resolved")
    print(f"holdout GWs: {holdout}")

    # Lazy import. Keep CLI startup cheap.
    from calibration import (match_calibration_summary, minutes_calibration_summary,
                             points_calibration_summary, write_markdown_report)

    points_pred = pd.DataFrame()
    match_pred = pd.DataFrame()
    minutes_pred = pd.DataFrame()

    recalib = None
    if args.recalib is not None:
        import json
        recalib = json.loads(args.recalib.read_text(encoding="utf-8"))
        print(f"recalib loaded from {args.recalib}")
    minutes_recalib = None
    if args.minutes_recalib is not None:
        import json
        minutes_recalib = json.loads(args.minutes_recalib.read_text(encoding="utf-8"))
        print(f"minutes_recalib loaded from {args.minutes_recalib}")

    if not args.skip_points:
        print("walk-forward points...")
        points_pred = walk_forward_points(holdout, history, fixtures, players, teams,
                                          recalib=recalib)
        if not points_pred.empty:
            points_pred.to_csv(OUT_DIR / "points_pred.csv", index=False)
            print(f"  saved {len(points_pred)} rows -> points_pred.csv")

    if not args.skip_match:
        print("walk-forward match...")
        match_pred = walk_forward_match(holdout, fixtures, history, teams)
        if not match_pred.empty:
            match_pred.to_csv(OUT_DIR / "match_pred.csv", index=False)
            print(f"  saved {len(match_pred)} rows -> match_pred.csv")

    if not args.skip_minutes:
        print("walk-forward minutes...")
        minutes_pred = walk_forward_minutes(holdout, history, fixtures, players, teams,
                                            recalib=minutes_recalib)
        if not minutes_pred.empty:
            minutes_pred.to_csv(OUT_DIR / "minutes_pred.csv", index=False)
            print(f"  saved {len(minutes_pred)} rows -> minutes_pred.csv")

    points_cal = points_calibration_summary(points_pred) if not points_pred.empty else pd.DataFrame()
    match_cal = match_calibration_summary(match_pred) if not match_pred.empty else pd.DataFrame()
    minutes_cal = minutes_calibration_summary(minutes_pred) if not minutes_pred.empty else pd.DataFrame()
    if not points_cal.empty:
        points_cal.to_csv(OUT_DIR / "points_calibration.csv", index=False)
    if not match_cal.empty:
        match_cal.to_csv(OUT_DIR / "match_calibration.csv", index=False)
    if not minutes_cal.empty:
        minutes_cal.to_csv(OUT_DIR / "minutes_calibration.csv", index=False)

    write_markdown_report(OUT_DIR / "report.md", holdout, points_cal, match_cal,
                          minutes_cal, points_pred, match_pred, minutes_pred)
    print(f"report -> {OUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
