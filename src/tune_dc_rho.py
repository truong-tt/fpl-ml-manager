"""Grid-search Dixon-Coles rho.

DC tau correction redistributes joint mass between (0,0)<->(0,1) and (1,0)<->(1,1)
but EXACTLY conserves row + column marginals. Algebra: gain at M[0,0] =
-lh*la*rho * exp(-lh-la) cancels loss at M[0,1] = +lh*la*rho * exp(-lh-la);
same for (1,0)<->(1,1). So clean-sheet probs (column / row sums) are
mathematically invariant to rho — earlier cs_brier-based tuning was a no-op,
returning identical numbers across grid.

rho only moves joint PMF (which scoreline most likely). Score with joint
negative log-likelihood on actual outcomes — only metric on which rho is
identifiable.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import train_match_model
from backtest import _resolve_holdout, walk_forward_match

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_RHOS = [-0.20, -0.15, -0.10, -0.05, 0.0]


def _joint_score_nll(pred: pd.DataFrame) -> float:
    """Mean -log P(gh, ga | lh, la) over fixtures using DC-corrected score matrix.

    Walk-forward `pred` carries lambda_h/lambda_a/gh/ga. Re-build joint score
    matrix at current module DC_RHO + read off actual cell. Only metric where
    rho is identifiable.
    """
    if pred.empty:
        return float("inf")
    lls: list[float] = []
    for _, r in pred.iterrows():
        M = train_match_model.score_matrix(float(r["lambda_h"]), float(r["lambda_a"]))
        gh, ga = int(r["gh"]), int(r["ga"])
        gh = min(gh, M.shape[0] - 1)
        ga = min(ga, M.shape[1] - 1)
        p = float(M[gh, ga])
        lls.append(-np.log(max(p, 1e-12)))
    return float(np.mean(lls))


def _coverage_low(pred: pd.DataFrame) -> float:
    """P(gh+ga <= 1) under DC-joint vs empirical. Sanity: correction shifting
    low-score corners. Returns mean predicted."""
    if pred.empty:
        return float("nan")
    vals: list[float] = []
    for _, r in pred.iterrows():
        M = train_match_model.score_matrix(float(r["lambda_h"]), float(r["lambda_a"]))
        vals.append(float(M[0, 0] + M[0, 1] + M[1, 0]))
    return float(np.mean(vals))


def grid_search(rhos: list[float], holdout: list[int],
                fixtures: pd.DataFrame, history: pd.DataFrame,
                teams: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward per rho via DC_RHO monkeypatch. Sort by joint score NLL.

    Walk-forward only needs to run once for lambdas (not function of rho), but
    walk_forward_match recomputes cs_h_p/cs_a_p per pass too. Re-run per rho
    keeps signature simple; cost = XGBoost retrain per holdout x rho combo.
    """
    rows: list[dict] = []
    original = train_match_model.DC_RHO
    try:
        for rho in rhos:
            train_match_model.DC_RHO = float(rho)
            pred = walk_forward_match(holdout, fixtures, history, teams)
            if pred.empty:
                rows.append({"rho": rho, "joint_nll": None, "p_low_score": None,
                             "n_fixtures": 0})
                continue
            nll = _joint_score_nll(pred)
            p_low = _coverage_low(pred)
            rows.append({"rho": rho, "joint_nll": nll, "p_low_score": p_low,
                         "n_fixtures": len(pred)})
            print(f"  rho={rho:+.2f}: joint_nll={nll:.4f}  p_low={p_low:.4f}")
    finally:
        train_match_model.DC_RHO = original
    return pd.DataFrame(rows).sort_values("joint_nll", na_position="last").reset_index(drop=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid search DC rho. Walk-forward joint score NLL")
    p.add_argument("--k", type=int, default=5, help="Trailing finished GWs. Default 5")
    p.add_argument("--start", type=int, default=None, help="Holdout start GW. Inclusive")
    p.add_argument("--end", type=int, default=None, help="Holdout end GW. Inclusive")
    p.add_argument("--rhos", type=float, nargs="+", default=DEFAULT_RHOS,
                   help="Grid of rho values. Default: -0.20 -0.15 -0.10 -0.05 0.0")
    p.add_argument("--out", type=Path,
                   default=DATA_DIR / "processed" / "backtest" / "dc_rho_grid.csv",
                   help="CSV output. Default: data/processed/backtest/dc_rho_grid.csv")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    fixtures = pd.read_csv(DATA_DIR / "fixtures.csv")
    history = pd.read_csv(DATA_DIR / "history.csv")
    teams = pd.read_csv(DATA_DIR / "teams.csv")
    holdout = _resolve_holdout(fixtures, args.k, args.start, args.end)
    if not holdout:
        raise RuntimeError("no holdout GWs resolved")
    print(f"holdout GWs: {holdout}")
    print(f"rho grid: {args.rhos}")
    print(f"current DC_RHO = {train_match_model.DC_RHO}")
    print()

    table = grid_search(args.rhos, holdout, fixtures, history, teams)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print()
    print(table.to_string(index=False))
    print()
    if not table.empty and table["joint_nll"].notna().any():
        winner = table.iloc[0]
        print(f"winner: rho={winner['rho']:+.2f} (joint_nll={winner['joint_nll']:.4f})")
        print(f"update train_match_model.DC_RHO = {winner['rho']}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
