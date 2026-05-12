"""Calibration audit for walk-forward preds from backtest.py.

Two model families, two test families:

- Points (quantile reg). Per-pos empirical coverage P(y <= q_alpha) vs nominal
  alpha. Pinball loss per quantile. Well-calibrated q90 = coverage 0.90.
  Consistent under-coverage = upper tail too tight (q90 too low).
- Match (Poisson + DC). Mean Poisson NLL, MAE on goals, Brier on CS probs,
  coverage check on CS rates.

No plotting deps. Outputs = CSVs + markdown table strings consumed by report
writer in backtest.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln

POS_NAME = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
QUANTILES = [0.10, 0.50, 0.90]
QCOLS = {0.10: "q10_pred", 0.50: "q50_pred", 0.90: "q90_pred"}


def _pinball(y: np.ndarray, q_pred: np.ndarray, alpha: float) -> float:
    """Pinball / check loss. Lower better. Match reg:quantileerror objective."""
    diff = y - q_pred
    return float(np.mean(np.where(diff >= 0, alpha * diff, (alpha - 1) * diff)))


def _points_rows(pred: pd.DataFrame, scope: str) -> list[dict]:
    """Per-(pos, quantile) calibration rows. Tag with scope ('all' or 'played')."""
    rows: list[dict] = []
    for pos in sorted(pred["pos_id"].dropna().unique().astype(int)):
        sub = pred[pred["pos_id"] == pos]
        n = len(sub)
        if n == 0:
            continue
        for alpha in QUANTILES:
            qcol = QCOLS[alpha]
            cov = float(np.mean(sub["y"].values <= sub[qcol].values))
            pin = _pinball(sub["y"].values.astype(float),
                           sub[qcol].values.astype(float), alpha)
            rows.append({
                "scope": scope,
                "pos": POS_NAME.get(int(pos), str(pos)),
                "alpha": alpha,
                "n": n,
                "coverage": cov,
                "coverage_gap": cov - alpha,
                "pinball_loss": pin,
            })
    return rows


def points_calibration_summary(pred: pd.DataFrame) -> pd.DataFrame:
    """Per-(pos, quantile) coverage + pinball in two scopes:

    - all. Every (player, GW) row including DNPs. y=0 inflates lower-tail cov.
    - played. minutes > 0. Production-conditional calibration.
    """
    if pred.empty:
        return pd.DataFrame()
    rows = _points_rows(pred, "all")
    if "minutes" in pred.columns:
        played = pred[pred["minutes"] > 0]
        if not played.empty:
            rows += _points_rows(played, "played")
    return pd.DataFrame(rows)


def points_overall(pred: pd.DataFrame) -> pd.DataFrame:
    """Pos-pooled coverage + pinball per quantile. all + played scopes."""
    if pred.empty:
        return pd.DataFrame()
    out: list[dict] = []
    scopes = [("all", pred)]
    if "minutes" in pred.columns:
        scopes.append(("played", pred[pred["minutes"] > 0]))
    for scope, sub in scopes:
        if sub.empty:
            continue
        for alpha in QUANTILES:
            qcol = QCOLS[alpha]
            cov = float(np.mean(sub["y"].values <= sub[qcol].values))
            pin = _pinball(sub["y"].values.astype(float),
                           sub[qcol].values.astype(float), alpha)
            out.append({"scope": scope, "alpha": alpha, "n": len(sub),
                        "coverage": cov, "coverage_gap": cov - alpha,
                        "pinball_loss": pin})
    return pd.DataFrame(out)


def combined_points_calibration_summary(pred: pd.DataFrame) -> pd.DataFrame:
    """Coverage of (q*_pred + bonus_q*_pred) vs y_total = y + bonus_actual.

    End-to-end audit of decoupled heads. Validates BONUS_BLEND=1.0. Empty if
    bonus columns absent (older predictions).
    """
    needed = {"bonus_q10_pred", "bonus_q50_pred", "bonus_q90_pred", "y_total"}
    if pred.empty or not needed.issubset(pred.columns):
        return pd.DataFrame()
    df = pred.copy()
    for q in ("q10_pred", "q50_pred", "q90_pred"):
        df[f"comb_{q}"] = df[q].astype(float) + df[f"bonus_{q}"].astype(float)
    rows: list[dict] = []
    for scope_name, sub in (("all", df), ("played", df[df.get("minutes", 0) > 0])):
        if sub.empty:
            continue
        for alpha in QUANTILES:
            qcol = f"comb_{QCOLS[alpha]}"
            cov = float(np.mean(sub["y_total"].values <= sub[qcol].values))
            pin = _pinball(sub["y_total"].values.astype(float),
                           sub[qcol].values.astype(float), alpha)
            rows.append({"scope": scope_name, "alpha": alpha, "n": len(sub),
                         "coverage": cov, "coverage_gap": cov - alpha,
                         "pinball_loss": pin})
    return pd.DataFrame(rows)


def _poisson_nll(y: np.ndarray, lam: np.ndarray, eps: float = 1e-9) -> float:
    """Mean Poisson NLL. gammaln for log(y!)."""
    lam = np.maximum(lam, eps)
    return float(np.mean(lam - y * np.log(lam) + gammaln(y + 1.0)))


def _auc_played(p: np.ndarray, y: np.ndarray) -> float:
    """ROC-AUC via Mann-Whitney U. No sklearn dep. 0.5 on class collapse."""
    pos = p[y == 1]
    neg = p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    n_pos, n_neg = len(pos), len(neg)
    ranks = pd.Series(np.concatenate([pos, neg])).rank().values
    rank_pos = ranks[:n_pos].sum()
    u = rank_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def minutes_calibration_summary(pred: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Calibration table for minutes / 90.

    Long-form rows tagged in `metric` col:

    - metric='summary': aggregate per scope (all / per pos). MAE on minutes/90 +
      binary-played AUC + Brier. mins_pred treated as P(played).
    - metric='bin': reliability bin row. Actual `played` rate per decile of
      mins_pred. Calibration plot / coverage gap check.
    """
    if pred.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    mins_norm_actual = (pred["mins_actual"].clip(upper=90) / 90.0).clip(0.0, 1.0).values
    mins_pred = pred["mins_pred"].values.astype(float)
    played = pred["played"].astype(int).values
    rows.append({
        "scope": "ALL",
        "metric": "summary",
        "n": len(pred),
        "mae_mins90": float(np.mean(np.abs(mins_pred - mins_norm_actual))),
        "brier_played": float(np.mean((mins_pred - played) ** 2)),
        "auc_played": _auc_played(mins_pred, played),
        "played_actual_rate": float(played.mean()),
        "played_pred_rate": float(mins_pred.mean()),
    })
    for pos in sorted(pred["pos_id"].dropna().unique().astype(int)):
        sub = pred[pred["pos_id"] == pos]
        if sub.empty:
            continue
        sn = (sub["mins_actual"].clip(upper=90) / 90.0).clip(0.0, 1.0).values
        sp = sub["mins_pred"].values.astype(float)
        sy = sub["played"].astype(int).values
        rows.append({
            "scope": POS_NAME.get(int(pos), str(pos)),
            "metric": "summary",
            "n": len(sub),
            "mae_mins90": float(np.mean(np.abs(sp - sn))),
            "brier_played": float(np.mean((sp - sy) ** 2)),
            "auc_played": _auc_played(sp, sy),
            "played_actual_rate": float(sy.mean()),
            "played_pred_rate": float(sp.mean()),
        })

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(mins_pred, edges[1:-1]), 0, n_bins - 1)
    for b in range(n_bins):
        m = (idx == b)
        if not m.any():
            continue
        rows.append({
            "scope": "ALL",
            "metric": "bin",
            "n": int(m.sum()),
            "bin_low": float(edges[b]),
            "bin_high": float(edges[b + 1]),
            "pred_mean": float(mins_pred[m].mean()),
            "played_actual": float(played[m].mean()),
            "gap": float(mins_pred[m].mean() - played[m].mean()),
        })
    return pd.DataFrame(rows)


def match_calibration_summary(pred: pd.DataFrame) -> pd.DataFrame:
    """Per-side Poisson NLL + goal MAE + CS Brier + CS coverage."""
    if pred.empty:
        return pd.DataFrame()
    rows = []
    for side, lam_col, y_col, cs_p_col, cs_a_col in (
        ("home", "lambda_h", "gh", "cs_h_p", "cs_h_actual"),
        ("away", "lambda_a", "ga", "cs_a_p", "cs_a_actual"),
    ):
        lam = pred[lam_col].values.astype(float)
        y = pred[y_col].values.astype(float)
        nll = _poisson_nll(y, lam)
        mae = float(np.mean(np.abs(lam - y)))
        cs_p = pred[cs_p_col].values.astype(float)
        cs_a = pred[cs_a_col].values.astype(float)
        brier = float(np.mean((cs_p - cs_a) ** 2))
        cs_pred_rate = float(np.mean(cs_p))
        cs_actual_rate = float(np.mean(cs_a))
        rows.append({
            "side": side,
            "n": len(pred),
            "poisson_nll": nll,
            "goal_mae": mae,
            "cs_brier": brier,
            "cs_pred_rate": cs_pred_rate,
            "cs_actual_rate": cs_actual_rate,
            "cs_rate_gap": cs_pred_rate - cs_actual_rate,
        })
    return pd.DataFrame(rows)


def _md_table(df: pd.DataFrame, fmt: dict[str, str] | None = None) -> str:
    if df.empty:
        return "_(empty)_"
    fmt = fmt or {}
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines = [head, sep]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if pd.isna(v):
                cells.append("-")
            elif c in fmt:
                cells.append(format(v, fmt[c]))
            elif isinstance(v, (int, np.integer)):
                cells.append(str(int(v)))
            elif isinstance(v, (float, np.floating)):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_markdown_report(path: Path, holdout: list[int],
                          points_cal: pd.DataFrame, match_cal: pd.DataFrame,
                          minutes_cal: pd.DataFrame,
                          points_pred: pd.DataFrame, match_pred: pd.DataFrame,
                          minutes_pred: pd.DataFrame) -> None:
    """Write single human-readable backtest report to path."""
    lines = [
        "# Walk-Forward Backtest Report",
        "",
        f"- Holdout GWs: {holdout}",
        f"- Points predictions: {len(points_pred)} (player, GW) rows",
        f"- Match predictions: {len(match_pred)} fixtures",
        f"- Minutes predictions: {len(minutes_pred)} (player, GW) rows",
        "",
        "## Points calibration (per position × quantile)",
        "",
        "Coverage = empirical P(y ≤ q_pred). Well-calibrated → coverage ≈ alpha. ",
        "Negative `coverage_gap` at q90 means upper tail is under-predicted (boom misses).",
        "",
    ]
    if not points_cal.empty:
        for scope in ("all", "played"):
            sub = points_cal[points_cal["scope"] == scope]
            if sub.empty:
                continue
            label = ("Every (player, GW) row (DNPs included)"
                     if scope == "all" else "Played-only (minutes > 0)")
            lines += [f"### Scope: {scope} — {label}", ""]
            lines.append(_md_table(sub.drop(columns="scope")))
            lines.append("")
        lines += ["### Position-pooled overall", ""]
        lines.append(_md_table(points_overall(points_pred)))
    else:
        lines.append("_no points predictions_")

    combined = combined_points_calibration_summary(points_pred) if not points_pred.empty else pd.DataFrame()
    if not combined.empty:
        lines += ["", "## Combined points + bonus calibration",
                  "",
                  "Coverage of `q*_pred + bonus_q*_pred` vs `total_points`. ",
                  "End-to-end check of decoupled heads under BONUS_BLEND=1.0.",
                  ""]
        lines.append(_md_table(combined))

    lines += ["", "## Minutes-model calibration", ""]
    if not minutes_cal.empty:
        summary = minutes_cal[minutes_cal["metric"] == "summary"]
        if not summary.empty:
            lines.append(_md_table(summary.drop(columns=["metric"])))
        bins = minutes_cal[minutes_cal["metric"] == "bin"]
        if not bins.empty:
            lines += ["", "### Reliability bins (predicted vs. actual played rate)", ""]
            keep = ["bin_low", "bin_high", "n", "pred_mean", "played_actual", "gap"]
            lines.append(_md_table(bins[keep]))
        lines += [
            "",
            "- `mae_mins90` = mean absolute error on minutes/90 (both ∈ [0, 1]).",
            "- `auc_played` = ROC-AUC for predicting played vs. DNP from the minutes head.",
            "- `brier_played` = mean squared error vs binary `played`.",
            "- Reliability `gap` > 0 in a bin ⇒ over-predicting playing time at that level.",
        ]
    else:
        lines.append("_no minutes predictions_")

    lines += ["", "## Match calibration", ""]
    if not match_cal.empty:
        lines.append(_md_table(match_cal))
        lines += [
            "",
            "- `poisson_nll` = mean per-fixture negative log-likelihood; lower better.",
            "- `goal_mae` = mean absolute error on goals.",
            "- `cs_brier` = mean squared error of predicted clean-sheet probability vs actual.",
            "- `cs_rate_gap` > 0 → model over-predicts clean sheets on average.",
        ]
    else:
        lines.append("_no match predictions_")

    lines += [
        "",
        "## Reading guide",
        "",
        "1. **Coverage gaps**: if `q90` coverage_gap < -0.05, ceiling predictions",
        "   under-fire — boom GWs are missed. Apply isotonic recalibration on residuals.",
        "2. **Position skew**: large gaps in one position only point to data scarcity",
        "   (FWD set is smallest at ~3k rows) or feature inadequacy for that position.",
        "3. **Match `cs_rate_gap`**: marginal Poisson calibration. CS = P(opp goals = 0).",
        "   Tune λ-head hyperparams if persistent positive/negative bias across seasons.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
