"""Per-pos isotonic recalib for minutes / 90 head. Closes 7.4 reliability gap."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_PRED = DATA_DIR / "processed" / "backtest" / "minutes_pred.csv"
DEFAULT_OUT = DATA_DIR / "minutes_recalib.json"
POSITIONS = [1, 2, 3, 4]
MIN_ROWS = 200


def fit_isotonic(p_pred: np.ndarray, played: np.ndarray) -> list[list[float]]:
    """IsotonicRegression(clip, [0, 1]). Return knot pairs [[x, y], ...]."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_pred, played)
    xs = iso.X_thresholds_.astype(float).tolist()
    ys = iso.y_thresholds_.astype(float).tolist()
    return [list(p) for p in zip(xs, ys)]


def fit_minutes_recalib(pred: pd.DataFrame) -> dict[str, list[list[float]]]:
    """Fit isotonic per pos_id. Skip pos with < MIN_ROWS rows."""
    coef: dict[str, list[list[float]]] = {}
    for pos in POSITIONS:
        sub = pred[pred["pos_id"] == pos]
        if len(sub) < MIN_ROWS:
            continue
        p = sub["mins_pred"].values.astype(float)
        y = sub["played"].astype(int).values.astype(float)
        coef[str(pos)] = fit_isotonic(p, y)
    return coef


def save_recalib(coef: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(coef, indent=2), encoding="utf-8")


def load_recalib(path: Path = DEFAULT_OUT) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def apply_recalib(coef: dict, pos_id: np.ndarray, p_pred: np.ndarray) -> np.ndarray:
    """Per-pos knots via np.interp. Unfitted positions pass through. Clip [0, 1]."""
    out = p_pred.astype(float).copy()
    for pos in POSITIONS:
        knots = coef.get(str(pos))
        if not knots:
            continue
        mask = (pos_id == pos)
        if not mask.any():
            continue
        xs = np.array([k[0] for k in knots], dtype=float)
        ys = np.array([k[1] for k in knots], dtype=float)
        out[mask] = np.interp(out[mask], xs, ys)
    return np.clip(out, 0.0, 1.0)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit per-pos isotonic recalib on walk-forward minutes preds")
    p.add_argument("--pred", type=Path, default=DEFAULT_PRED,
                   help="Walk-forward minutes preds CSV. Default: backtest output")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="Output JSON path. Default: data/minutes_recalib.json")
    p.add_argument("--fit-end-gw", type=int, default=None,
                   help="Fit only rows with gw <= fit_end_gw. Held-out tail evaluates")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    pred = pd.read_csv(args.pred)
    if args.fit_end_gw is not None:
        pred = pred[pred["gw"] <= args.fit_end_gw]
        print(f"fit set: gw <= {args.fit_end_gw}, {len(pred)} rows")
    else:
        print(f"fit set: all of {args.pred}, {len(pred)} rows")
    coef = fit_minutes_recalib(pred)
    save_recalib(coef, args.out)
    print(f"saved -> {args.out}")
    for pos, knots in coef.items():
        print(f"  pos={pos} knots={len(knots)} "
              f"x_range=[{knots[0][0]:.3f}, {knots[-1][0]:.3f}] "
              f"y_range=[{knots[0][1]:.3f}, {knots[-1][1]:.3f}]")


if __name__ == "__main__":
    main()
