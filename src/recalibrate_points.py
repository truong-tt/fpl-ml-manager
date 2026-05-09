"""Per-(pos, quantile) isotonic recalib for points head.

Replaces affine `a + b * q_pred` with non-parametric monotone curve. Affine left
tail residual non-linearity (q90 still under-fired on booms after slope correction).
Same family as minutes head 4.7 isotonic.

Fit:
1. Played-only rows (engine multiplies raw quantile by mins_pred → recalibrating
   on DNP-inflated dist double-counts availability multiplier).
2. Equal-frequency bin q_pred into N bins. Per bin: empirical alpha-quantile of
   y (Bayes-optimal estimate of conditional alpha-quantile in bin).
3. sklearn IsotonicRegression on (bin_center, bin_alpha_quantile).
4. Persist knots JSON. Inference = linear-interp between knots.

Non-crossing re-enforced row-wise (sort) at apply time. Three quantile recalibs
fit independently.

Affine fallback for any (pos, alpha) cell with < MIN_ROWS — isotonic on small
samples = high-variance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_PRED = DATA_DIR / "processed" / "backtest" / "points_pred.csv"
DEFAULT_OUT = DATA_DIR / "points_recalib.json"
QUANTILES = [0.10, 0.50, 0.90]
QCOLS = {0.10: "q10_pred", 0.50: "q50_pred", 0.90: "q90_pred"}
POSITIONS = [1, 2, 3, 4]

MIN_ROWS_ISOTONIC = 400  # below = isotonic over-fits empty bins
N_BINS_DEFAULT = 30
B_MIN, B_MAX = 0.1, 5.0  # affine fallback slope bounds
# Coverage-gap gate. Recalib kept only if |gap_cal| + EPS < |gap_raw|, AND
# pinball doesn't regress. Pinball-only gate accepted MID q50 recalib that
# halved pinball but doubled coverage gap (+0.04 → +0.08). Coverage is the
# audited metric — gate on it.
COVERAGE_GAP_EPS = 0.005
PINBALL_REGRESS_TOL = 0.02  # accept ≤2% pinball regression if coverage improves


def pinball_loss(y: np.ndarray, q: np.ndarray, alpha: float) -> float:
    """Pinball / check loss. Match XGBoost reg:quantileerror objective."""
    diff = y - q
    return float(np.mean(np.where(diff >= 0, alpha * diff, (alpha - 1) * diff)))


def fit_affine(y: np.ndarray, q_pred: np.ndarray, alpha: float) -> tuple[float, float]:
    """SLSQP min pinball(y, a + b * q_pred). Slope bounded [B_MIN, B_MAX]."""
    if len(y) < 20:
        return 0.0, 1.0

    def obj(params: np.ndarray) -> float:
        a, b = params
        return pinball_loss(y, a + b * q_pred, alpha)

    res = minimize(obj, x0=np.array([0.0, 1.0]), method="SLSQP",
                   bounds=[(-50.0, 50.0), (B_MIN, B_MAX)],
                   options={"ftol": 1e-6, "maxiter": 200})
    return float(res.x[0]), float(res.x[1])


def fit_isotonic_quantile(q_pred: np.ndarray, y: np.ndarray, alpha: float,
                          n_bins: int = N_BINS_DEFAULT) -> list[list[float]]:
    """Equal-freq bin q_pred → per-bin empirical alpha-quantile of y → isotonic.

    Returns knots [[x, y], ...] for np.interp at apply time.
    """
    n = len(q_pred)
    if n < MIN_ROWS_ISOTONIC:
        return []

    df = pd.DataFrame({"q": q_pred.astype(float), "y": y.astype(float)})
    bins = pd.qcut(df["q"], q=min(n_bins, n // 8), duplicates="drop", labels=False)
    df = df.assign(bin=bins).dropna(subset=["bin"])
    if df["bin"].nunique() < 5:
        return []
    grp = df.groupby("bin", observed=True).agg(
        q_center=("q", "mean"),
        y_alpha=("y", lambda v: float(np.quantile(v, alpha))),
        n=("y", "size"),
    ).sort_values("q_center")
    if len(grp) < 5:
        return []

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(grp["q_center"].values, grp["y_alpha"].values,
            sample_weight=grp["n"].values)
    xs = iso.X_thresholds_.astype(float).tolist()
    ys = iso.y_thresholds_.astype(float).tolist()
    return [list(p) for p in zip(xs, ys)]


def fit_points_recalib(pred: pd.DataFrame, played_only: bool = True
                       ) -> dict[str, dict[str, dict]]:
    """Per (pos, alpha): isotonic if rows >= MIN_ROWS_ISOTONIC, else affine fallback.

    Identity-gate: candidate spec dropped if pinball not beat raw by
    PINBALL_IMPROVE_MIN. Raw passthrough at apply time.

    Schema: {pos_id: {qNN: {type: iso|affine, knots|ab: ...}}}
    """
    if played_only and "minutes" in pred.columns:
        pred = pred[pred["minutes"] > 0]
    coef: dict[str, dict[str, dict]] = {}
    for pos in POSITIONS:
        sub = pred[pred["pos_id"] == pos]
        if sub.empty:
            continue
        coef[str(pos)] = {}
        for alpha in QUANTILES:
            qcol = QCOLS[alpha]
            y = sub["y"].values.astype(float)
            qp = sub[qcol].values.astype(float)
            knots = fit_isotonic_quantile(qp, y, alpha)
            key = f"q{int(alpha * 100):02d}"
            if knots:
                spec = {"type": "iso", "knots": knots}
            else:
                a, b = fit_affine(y, qp, alpha)
                spec = {"type": "affine", "ab": [a, b]}
            calibrated = _apply_one(spec, qp)
            raw_gap = abs(float((y <= qp).mean()) - alpha)
            cal_gap = abs(float((y <= calibrated).mean()) - alpha)
            raw_pl = pinball_loss(y, qp, alpha)
            cal_pl = pinball_loss(y, calibrated, alpha)
            coverage_better = cal_gap + COVERAGE_GAP_EPS < raw_gap
            pinball_acceptable = (raw_pl <= 0 or
                                   (cal_pl - raw_pl) / raw_pl <= PINBALL_REGRESS_TOL)
            if not (coverage_better and pinball_acceptable):
                continue  # gated. Raw passthrough.
            coef[str(pos)][key] = spec
    return coef


def save_recalib(coef: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(coef, indent=2), encoding="utf-8")


def load_recalib(path: Path = DEFAULT_OUT) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _apply_one(spec: dict, vals: np.ndarray) -> np.ndarray:
    """Apply one (pos, alpha) recalib spec. Supports legacy [a, b] lists."""
    if isinstance(spec, list) and len(spec) == 2:  # legacy affine list
        a, b = float(spec[0]), float(spec[1])
        return a + b * vals
    t = spec.get("type")
    if t == "affine":
        a, b = float(spec["ab"][0]), float(spec["ab"][1])
        return a + b * vals
    if t == "iso":
        knots = spec["knots"]
        xs = np.array([k[0] for k in knots], dtype=float)
        ys = np.array([k[1] for k in knots], dtype=float)
        return np.interp(vals, xs, ys)
    return vals


def apply_recalib(coef: dict, pos_id: np.ndarray, quantiles: pd.DataFrame
                  ) -> pd.DataFrame:
    """Per-row pos x quantile recalib. Re-enforce non-crossing."""
    out = quantiles.copy()
    for pos in POSITIONS:
        pos_coef = coef.get(str(pos))
        if pos_coef is None:
            continue
        mask = (pos_id == pos)
        if not mask.any():
            continue
        for col in ("q10", "q50", "q90"):
            spec = pos_coef.get(col)
            if spec is None:
                continue
            out.loc[mask, col] = _apply_one(spec, out.loc[mask, col].values.astype(float))
    vals = out[["q10", "q50", "q90"]].values.copy()
    vals.sort(axis=1)
    out[["q10", "q50", "q90"]] = vals
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit isotonic recalib on walk-forward preds")
    p.add_argument("--pred", type=Path, default=DEFAULT_PRED,
                   help="Walk-forward preds CSV. Default: backtest output")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="Output JSON. Default: data/points_recalib.json")
    p.add_argument("--fit-end-gw", type=int, default=None,
                   help="Fit only rows with gw <= fit_end_gw")
    p.add_argument("--include-dnp", action="store_true",
                   help="Include DNP (minutes=0) rows. Default: played-only")
    return p.parse_args()


def _summarise(coef: dict) -> None:
    for pos, qmap in coef.items():
        for q, spec in qmap.items():
            t = spec.get("type") if isinstance(spec, dict) else "affine_legacy"
            if t == "iso":
                k = spec["knots"]
                print(f"  pos={pos} {q} iso knots={len(k)} "
                      f"x=[{k[0][0]:.2f}, {k[-1][0]:.2f}] "
                      f"y=[{min(p[1] for p in k):.2f}, {max(p[1] for p in k):.2f}]")
            elif t == "affine":
                a, b = spec["ab"]
                print(f"  pos={pos} {q} affine a={a:+.3f} b={b:.3f}")


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
    _summarise(coef)


if __name__ == "__main__":
    main()
