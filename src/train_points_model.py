"""Per-pos XGBoost quantile regressors. q10/q50/q90 x {GK, DEF, MID, FWD}."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import xgboost as xgb

from features import build_match_features, build_player_features, points_feature_cols

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
QUANTILES = [0.10, 0.50, 0.90]
POSITIONS = [1, 2, 3, 4]


def _model_file(q: float, pos: int) -> str:
    return f"xgb_points_q{int(q * 100):02d}_p{pos}.json"


def _pos_feature_cols() -> list[str]:
    """Drop pos one-hots. Constant within per-pos model."""
    return [c for c in points_feature_cols() if not c.startswith("pos_")]


# Per-feature monotone direction. +1 = output non-decreasing in feature, 0 = no
# constraint. Cross-position-safe set: form proxies (+), set-piece flags (+),
# fixture context (+/-). Skip per-action rolls (saves/cbi/tkl) since direction
# differs by pos. Reduces overfit on small per-pos subsets (FWD ~3k, GK ~1k).
_MONO_DIR = {
    "is_home": 1,
    "is_pen_taker": 1,
    "is_fk_taker": 1,
    "elo_gap": 1,
    "own_elo": 1,
    "opp_elo": -1,
    "opp_xga_5": 1,    # opp concedes more → outfield score more
    "own_lambda_for": 1,   # team scores more → outfield/GK pos varies; net + via attack share
    "own_lambda_against": -1,  # team concedes more → DEF/GK lose CS, MID/FWD unaffected
    "own_cs_p": 1,     # higher CS prob → DEF/GK boom; outfield neutral. Net +.
    "lag1_min": 1, "lag2_min": 1, "lag3_min": 1,
    "roll5_xg": 1, "roll5_xa": 1, "roll5_xgi": 1,
    "roll5_bps": 1, "roll5_ict": 1,
    "roll10_xg": 1, "roll10_xa": 1, "roll10_xgi": 1,
    "roll10_bps": 1, "roll10_ict": 1,
    "roll5_oxg": 1, "roll5_oxa": 1, "roll5_occ": 1,
    "roll5_otob": 1, "roll5_osh": 1, "roll5_odrib": 1,
    "roll10_oxg": 1, "roll10_oxa": 1, "roll10_occ": 1,
    "roll10_otob": 1, "roll10_osh": 1, "roll10_odrib": 1,
    "roll5_xg_share": 1, "roll5_xa_share": 1, "roll5_xgi_share": 1,
    "roll10_xg_share": 1, "roll10_xa_share": 1, "roll10_xgi_share": 1,
}


def _monotone_constraints(feat_cols: list[str]) -> str:
    """XGBoost-format string: '(0,1,0,...)'. 0 default. Match feat_cols order."""
    dirs = [_MONO_DIR.get(c, 0) for c in feat_cols]
    return "(" + ",".join(str(d) for d in dirs) + ")"


def _row_pos(df: pd.DataFrame) -> pd.Series:
    """Recover pos_id 1..4 from pos_{1..4} one-hots."""
    return df[[f"pos_{p}" for p in POSITIONS]].idxmax(axis=1).str.replace("pos_", "").astype(int)


def train_points_models() -> None:
    """Train + serialize 4 pos x 3 quantiles = 12 boosters under data/.

    Played-only filter. Head learns E[pts | played]. Engine gates w/ P(plays) +
    E[mins/90] from minutes head. Stops DNP mass collapsing q10 to ~0 (DEF q10
    coverage gap -0.094 pre-fix). Recalib already played-only — alignment
    preserved.
    """
    fx = pd.read_csv(DATA_DIR / "fixtures.csv")
    hist = pd.read_csv(DATA_DIR / "history.csv")
    players = pd.read_csv(DATA_DIR / "players.csv")
    teams = pd.read_csv(DATA_DIR / "teams.csv")

    fixture_feats = build_match_features(fx, hist, teams)
    train = build_player_features(hist, players, fixture_feats).dropna(subset=["target"])
    if "minutes" in train.columns:
        train = train[train["minutes"] > 0]
    if train.empty:
        return

    train["_pos"] = _row_pos(train)
    feat_cols = _pos_feature_cols()
    mono = _monotone_constraints(feat_cols)

    for pos in POSITIONS:
        sub = train[train["_pos"] == pos]
        if len(sub) < 200:
            continue
        X = sub[feat_cols].astype(float).fillna(0.0)
        y = sub["target"].astype(float)
        for q in QUANTILES:
            # Per-pos sets small (~3k FWD). Strong reg keeps q90 credible.
            # Monotone constraints stop overfitting on noisy small subsets.
            params = dict(objective="reg:quantileerror", quantile_alpha=q,
                          learning_rate=0.03, max_depth=3, subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=30,
                          reg_alpha=0.5, reg_lambda=2.0, verbosity=0,
                          monotone_constraints=mono)
            m = xgb.train(params, xgb.DMatrix(X, label=y), num_boost_round=400)
            m.save_model(DATA_DIR / _model_file(q, pos))


def load_points_models() -> dict[int, dict[float, xgb.Booster]] | None:
    """Load {pos: {q: booster}}. None if any file missing."""
    out: dict[int, dict[float, xgb.Booster]] = {}
    for pos in POSITIONS:
        out[pos] = {}
        for q in QUANTILES:
            p = DATA_DIR / _model_file(q, pos)
            if not p.exists():
                return None
            b = xgb.Booster()
            b.load_model(p)
            out[pos][q] = b
    return out


def predict_quantiles(
    models: dict[int, dict[float, xgb.Booster]], X: pd.DataFrame,
    apply_recalib: bool = True,
) -> pd.DataFrame:
    """Route row to pos booster. Enforce non-crossing q10<=q50<=q90.

    apply_recalib + data/points_recalib.json present → per-(pos, alpha) iso/affine
    map applied post-sort, before sanity clip.
    """
    feat_cols = _pos_feature_cols()
    out = pd.DataFrame(0.0, index=X.index, columns=["q10", "q50", "q90"])
    pos_series = _row_pos(X)
    Xf = X[feat_cols].astype(float).fillna(0.0)
    for pos in POSITIONS:
        mask = (pos_series == pos)
        if not mask.any() or pos not in models:
            continue
        dm = xgb.DMatrix(Xf.loc[mask])
        for q, m in models[pos].items():
            out.loc[mask, f"q{int(q * 100):02d}"] = m.predict(dm)
    vals = out[["q10", "q50", "q90"]].values
    vals = vals.copy()
    vals.sort(axis=1)
    out[["q10", "q50", "q90"]] = vals

    if apply_recalib:
        from recalibrate_points import apply_recalib as _apply, load_recalib
        coef = load_recalib()
        if coef is not None:
            out = _apply(coef, pos_series.values, out)

    # Sanity ceiling. Credible single-GW boom ~25 (hat-trick + assist + bonus).
    return out.clip(lower=-3.0, upper=25.0)


if __name__ == "__main__":
    train_points_models()
