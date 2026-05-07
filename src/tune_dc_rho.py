"""Grid-search Dixon-Coles ρ. Minimize walk-forward CS Brier from §7.3."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import train_match_model
from backtest import _resolve_holdout, walk_forward_match
from calibration import match_calibration_summary

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_RHOS = [-0.20, -0.15, -0.10, -0.05, 0.0]


def _score(pred: pd.DataFrame) -> tuple[float, float, float]:
    """Return (cs_brier, |cs_rate_gap|, poisson_nll) means. Lower better."""
    cal = match_calibration_summary(pred)
    if cal.empty:
        return float("inf"), float("inf"), float("inf")
    cs_brier = float(cal["cs_brier"].mean())
    cs_gap_abs = float(cal["cs_rate_gap"].abs().mean())
    nll = float(cal["poisson_nll"].mean())
    return cs_brier, cs_gap_abs, nll


def grid_search(rhos: list[float], holdout: list[int],
                fixtures: pd.DataFrame, history: pd.DataFrame,
                teams: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward per ρ via DC_RHO monkeypatch. Return table sorted by cs_brier."""
    rows: list[dict] = []
    original = train_match_model.DC_RHO
    try:
        for rho in rhos:
            train_match_model.DC_RHO = float(rho)
            pred = walk_forward_match(holdout, fixtures, history, teams)
            if pred.empty:
                rows.append({"rho": rho, "cs_brier": None, "cs_gap_abs": None,
                             "poisson_nll": None, "n_fixtures": 0})
                continue
            cs_brier, cs_gap_abs, nll = _score(pred)
            rows.append({"rho": rho, "cs_brier": cs_brier,
                         "cs_gap_abs": cs_gap_abs, "poisson_nll": nll,
                         "n_fixtures": len(pred)})
            print(f"  ρ={rho:+.2f}: cs_brier={cs_brier:.4f} "
                  f"|cs_gap|={cs_gap_abs:.4f} nll={nll:.4f}")
    finally:
        train_match_model.DC_RHO = original
    return pd.DataFrame(rows).sort_values("cs_brier", na_position="last").reset_index(drop=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid search Dixon-Coles ρ. Walk-forward CS Brier")
    p.add_argument("--k", type=int, default=5, help="Trailing finished GWs. Default 5")
    p.add_argument("--start", type=int, default=None, help="Holdout start GW. Inclusive")
    p.add_argument("--end", type=int, default=None, help="Holdout end GW. Inclusive")
    p.add_argument("--rhos", type=float, nargs="+", default=DEFAULT_RHOS,
                   help="Grid of ρ values. Default: -0.20 -0.15 -0.10 -0.05 0.0")
    p.add_argument("--out", type=Path,
                   default=DATA_DIR / "processed" / "backtest" / "dc_rho_grid.csv",
                   help="CSV output path. Default: data/processed/backtest/dc_rho_grid.csv")
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
    print(f"ρ grid: {args.rhos}")
    print(f"current DC_RHO = {train_match_model.DC_RHO}")
    print()

    table = grid_search(args.rhos, holdout, fixtures, history, teams)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print()
    print(table.to_string(index=False))
    print()
    if not table.empty and table["cs_brier"].notna().any():
        winner = table.iloc[0]
        print(f"winner: ρ={winner['rho']:+.2f} (cs_brier={winner['cs_brier']:.4f})")
        print(f"update train_match_model.DC_RHO = {winner['rho']}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
