"""Per-(position, quantile) affine recalibration for points model. Played-only pinball fit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_PRED = DATA_DIR / "processed" / "backtest" / "points_pred.csv"
DEFAULT_OUT = DATA_DIR / "points_recalib.json"
QUANTILES = [0.10, 0.50, 0.90]
QCOLS = {0.10: "q10_pred", 0.50: "q50_pred", 0.90: "q90_pred"}
POSITIONS = [1, 2, 3, 4]


def pinball_loss(y: np.ndarray, q: np.ndarray, alpha: float) -> float:
    """Pinball / check loss. Match XGBoost `reg:quantileerror` objective."""
    diff = y - q
    return float(np.mean(np.where(diff >= 0, alpha * diff, (alpha - 1) * diff)))


# Lower bound on slope. b=0 collapses recalib to constant, destroys booster
# per-row ranking signal optimizer variance + captaincy terms depend on.
# 0.1 keeps rank order intact. Marginal pinball-loss cost on flat-target quantiles.
B_MIN, B_MAX = 0.1, 5.0


def fit_affine(y: np.ndarray, q_pred: np.ndarray, alpha: float
               ) -> tuple[float, float]:
    """Return (a, b). b in [B_MIN, B_MAX]. Minimize pinball(y, a + b*q_pred)."""
    if len(y) < 20:
        return 0.0, 1.0

    def obj(params: np.ndarray) -> float:
        a, b = params
        return pinball_loss(y, a + b * q_pred, alpha)

    # SLSQP with explicit bounds. Nelder-Mead does not respect bounds.
    res = minimize(obj, x0=np.array([0.0, 1.0]), method="SLSQP",
                   bounds=[(-50.0, 50.0), (B_MIN, B_MAX)],
                   options={"ftol": 1e-6, "maxiter": 200})
    a, b = float(res.x[0]), float(res.x[1])
    return a, b


def fit_points_recalib(pred: pd.DataFrame, played_only: bool = True
                       ) -> dict[str, dict[str, list[float]]]:
    """Fit affine (a, b) per (pos, alpha). Return {pos_id: {"qNN": [a, b]}}."""
    if played_only and "minutes" in pred.columns:
        pred = pred[pred["minutes"] > 0]
    coef: dict[str, dict[str, list[float]]] = {}
    for pos in POSITIONS:
        sub = pred[pred["pos_id"] == pos]
        if sub.empty:
            continue
        coef[str(pos)] = {}
        for alpha in QUANTILES:
            qcol = QCOLS[alpha]
            a, b = fit_affine(sub["y"].values.astype(float),
                              sub[qcol].values.astype(float), alpha)
            coef[str(pos)][f"q{int(alpha * 100):02d}"] = [a, b]
    return coef


def save_recalib(coef: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(coef, indent=2), encoding="utf-8")


def load_recalib(path: Path = DEFAULT_OUT) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def apply_recalib(coef: dict, pos_id: np.ndarray, quantiles: pd.DataFrame
                  ) -> pd.DataFrame:
    """Apply (a, b) per row pos_id × quantile. Unfitted positions pass through. Re-enforce non-crossing."""
    out = quantiles.copy()
    for pos in POSITIONS:
        pos_coef = coef.get(str(pos))
        if pos_coef is None:
            continue
        mask = (pos_id == pos)
        if not mask.any():
            continue
        for col, key in (("q10", "q10"), ("q50", "q50"), ("q90", "q90")):
            ab = pos_coef.get(key)
            if ab is None:
                continue
            a, b = float(ab[0]), float(ab[1])
            out.loc[mask, col] = a + b * out.loc[mask, col].values
    # Re-enforce non-crossing after row-wise affine maps.
    vals = out[["q10", "q50", "q90"]].values.copy()
    vals.sort(axis=1)
    out[["q10", "q50", "q90"]] = vals
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit affine recalibration on walk-forward predictions")
    p.add_argument("--pred", type=Path, default=DEFAULT_PRED,
                   help="Walk-forward predictions CSV. Default: backtest output")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="Output JSON path. Default: data/points_recalib.json")
    p.add_argument("--fit-end-gw", type=int, default=None,
                   help="If set, fit only on rows with gw <= fit_end_gw. Held-out tail evaluates")
    p.add_argument("--include-dnp", action="store_true",
                   help="Include DNP (minutes=0) rows. Default uses played-only")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    pred = pd.read_csv(args.pred)
    if args.fit_end_gw is not None:
        pred = pred[pred["gw"] <= args.fit_end_gw]
        print(f"fit set: gw <= {args.fit_end_gw}, {len(pred)} rows")
    else:
        print(f"fit set: all of {args.pred}, {len(pred)} rows")
    coef = fit_points_recalib(pred, played_only=not args.include_dnp)
    save_recalib(coef, args.out)
    print(f"saved -> {args.out}")
    for pos, qmap in coef.items():
        for q, ab in qmap.items():
            print(f"  pos={pos} {q}: a={ab[0]:+.3f}  b={ab[1]:.3f}")


if __name__ == "__main__":
    main()
