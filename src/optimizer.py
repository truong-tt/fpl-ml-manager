"""MILP optimizer. 15-man squad + per-GW XI + captain. RHC transfer plan."""
from __future__ import annotations

from typing import Any, Callable

import pandas as pd
import pulp

SQUAD_COUNTS = {1: 2, 2: 5, 3: 5, 4: 3}
MIN_STARTERS = {1: 1, 2: 3, 3: 2, 4: 1}
SQUAD_SIZE, XI_SIZE, MAX_PER_CLUB = 15, 11, 3
# DEF booms (CS+goal) correlated with team performance. MID/FWD upside uncorrelated.
CAPTAIN_POSITIONS = {3, 4}

# RHC attenuation. Geometric discount γ^k over horizon — standard MPC form.
# γ=0.85 → 8-GW profile [1.00, 0.85, 0.72, 0.61, 0.52, 0.44, 0.38, 0.32].
# Long-tail fixture-swing info still feeds plan (Liverpool Apr-May, Arsenal
# run-in) but trusted less the further out we look. Hits cost NOT attenuated:
# -4 = -4 regardless of horizon distance.
RHC_DISCOUNT = 0.85
RHC_HORIZON = 8
DEFAULT_ATTENUATION = [RHC_DISCOUNT ** k for k in range(RHC_HORIZON)]


def _attenuation_weights(gws: list[int], att: list[float] | None) -> dict[int, float]:
    """Map each GW in horizon to attenuation weight. Pad / truncate to len(gws)."""
    profile = list(att) if att is not None else list(DEFAULT_ATTENUATION)
    if len(profile) < len(gws):
        profile = profile + [profile[-1]] * (len(gws) - len(profile))
    return {t: float(profile[k]) for k, t in enumerate(gws)}


def _gws(proj: pd.DataFrame) -> list[int]:
    """Sorted GW list from xp_{t} columns."""
    return sorted(int(c.split("_")[1]) for c in proj.columns if c.startswith("xp_"))


def _add_core(
    prob: pulp.LpProblem, proj: pd.DataFrame, ids: list[int],
    s: dict, c: dict, x_of: Callable, gws: list[int],
) -> None:
    """XI size, captain size, GK quota, formation min, captain-must-start, captain pos."""
    for t in gws:
        prob += pulp.lpSum(s[i][t] for i in ids) == XI_SIZE
        prob += pulp.lpSum(c[i][t] for i in ids) == 1
        prob += pulp.lpSum(s[i][t] for i in ids if proj.loc[i, "pos_id"] == 1) == 1
        for pos, mn in MIN_STARTERS.items():
            prob += pulp.lpSum(s[i][t] for i in ids if proj.loc[i, "pos_id"] == pos) >= mn
        for i in ids:
            prob += s[i][t] <= x_of(i, t)
            prob += c[i][t] <= s[i][t]
            if int(proj.loc[i, "pos_id"]) not in CAPTAIN_POSITIONS:
                prob += c[i][t] == 0


def _extract(
    proj: pd.DataFrame, ids: list[int], gws: list[int],
    x: dict, s: dict, c: dict, x_per_t: bool,
) -> tuple[set[int], set[int], int, int]:
    """Pull (squad_ids, xi_ids, captain_id, vice_id) for first GW in horizon."""
    t0 = gws[0]
    if x_per_t:
        squad = {i for i in ids if x[i][t0].varValue and x[i][t0].varValue > 0.5}
    else:
        squad = {i for i in ids if x[i].varValue and x[i].varValue > 0.5}
    xi = {i for i in squad if s[i][t0].varValue and s[i][t0].varValue > 0.5}
    cap = next((i for i in xi if c[i][t0].varValue and c[i][t0].varValue > 0.5), None)
    non_cap = [i for i in xi if i != cap and int(proj.loc[i, "pos_id"]) in CAPTAIN_POSITIONS]
    cap_col = f"cap_xp_{t0}" if f"cap_xp_{t0}" in proj.columns else f"xp_{t0}"
    vice = max(non_cap, key=lambda i: float(proj.loc[i, cap_col])) if non_cap else None
    return squad, xi, cap, vice


def solve_initial_squad(
    proj: pd.DataFrame, budget: float = 100.0,
    lambda_var: float = 0.02, lambda_eo: float = 0.0,
    bench_weight: float = 0.15, bank_penalty: float = 0.5,
    time_limit: int = 60,
    attenuation: list[float] | None = None,
) -> pd.DataFrame:
    """Cold-start 15-man squad + per-GW XI/captain over horizon.

    bank_penalty: flat EV cost per £1m of unspent budget. Implemented as a
    + bank_penalty * price coefficient in the objective (the additive
    `−bank_penalty * budget` term is constant and drops out). Early-season
    noisy projections leave premium picks only marginally above cheap ones
    in μ; a non-zero penalty breaks the tie by pushing the solver to spend
    the full budget rather than parking £10m+ in the bank. Set to 0.0 to
    recover the previous "≤ budget, no incentive to spend" behaviour.
    """
    if proj.empty:
        return pd.DataFrame()
    proj = proj.copy().set_index("id")
    gws = _gws(proj)
    if not gws:
        return pd.DataFrame()
    ids = list(proj.index)
    n_gw = len(gws)
    att = _attenuation_weights(gws, attenuation)

    prob = pulp.LpProblem("FPL_Init", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in ids}
    s = {i: {t: pulp.LpVariable(f"s_{i}_{t}", cat="Binary") for t in gws} for i in ids}
    c = {i: {t: pulp.LpVariable(f"c_{i}_{t}", cat="Binary") for t in gws} for i in ids}

    obj = 0
    for t in gws:
        w = att[t]
        for i in ids:
            xp = float(proj.loc[i, f"xp_{t}"])
            cap_xp = float(proj.loc[i, f"cap_xp_{t}"])
            var = float(proj.loc[i, f"var_{t}"])
            eo = float(proj.loc[i, "eo"])
            obj += w * (xp * s[i][t] + bench_weight * xp * (x[i] - s[i][t])
                        + cap_xp * c[i][t])
            obj += -(w * lambda_var * var / n_gw) * x[i]
            obj += (w * lambda_eo * xp * (1.0 - eo) / n_gw) * x[i]
    # Bank-leftover penalty (B). Constant `−bank_penalty * budget` dropped.
    if bank_penalty > 0.0:
        obj += bank_penalty * pulp.lpSum(proj.loc[i, "price"] * x[i] for i in ids)
    prob += obj

    prob += pulp.lpSum(x[i] for i in ids) == SQUAD_SIZE
    for pos, cnt in SQUAD_COUNTS.items():
        prob += pulp.lpSum(x[i] for i in ids if proj.loc[i, "pos_id"] == pos) == cnt
    for tm in proj["team_id"].unique():
        prob += pulp.lpSum(x[i] for i in ids if proj.loc[i, "team_id"] == tm) <= MAX_PER_CLUB
    prob += pulp.lpSum(proj.loc[i, "price"] * x[i] for i in ids) <= budget

    _add_core(prob, proj, ids, s, c, lambda i, t: x[i], gws)
    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit))
    if pulp.LpStatus[prob.status] not in ("Optimal", "Not Solved"):
        return pd.DataFrame()

    squad, xi, cap, vice = _extract(proj, ids, gws, x, s, c, x_per_t=False)
    out = proj.loc[list(squad)].reset_index()
    out["in_xi"] = out["id"].isin(xi).astype(int)
    out["is_captain"] = (out["id"] == cap).astype(int)
    out["is_vice"] = (out["id"] == vice).astype(int)
    return out


def solve_rhc_transfers(
    proj: pd.DataFrame, current_squad_ids: set[int],
    bank: float, free_transfers: int, lambda_var: float = 0.02,
    lambda_eo: float = 0.0, bench_weight: float = 0.15,
    hit_cost: float = 6.0, hit_cap_per_gw: int = 1,
    bank_transfer_weight: float = 0.3, time_limit: int = 60,
    attenuation: list[float] | None = None,
) -> dict[str, Any]:
    """Receding-horizon transfer planner. Return this-GW squad / XI / captain / transfers.

    hit_cost: pts deducted per hit. FPL rule = 4. Raise to discourage churn -
        prior runs took ~40 hits/season (-160 pts), top managers 4-8.
    hit_cap_per_gw: max hits per GW. Default 1 (= 1 extra transfer beyond
        FT, costing 4 pts). Hard ceiling stops the solver chasing 8-pt+
        marginal gains it can rarely realise.
    bank_transfer_weight: positive EV reward per saved transfer. FT roll-over
        carries option value (best-pick-next-week). Without this term sv=0
        always (no upside in solver, only constraint slack). Tune.
    """
    if proj.empty:
        return {"status": "empty"}
    proj = proj.copy().set_index("id")
    gws = _gws(proj)
    if not gws:
        return {"status": "no_horizon"}
    ids = list(proj.index)
    att = _attenuation_weights(gws, attenuation)

    prob = pulp.LpProblem("FPL_RHC", pulp.LpMaximize)
    x = {i: {t: pulp.LpVariable(f"x_{i}_{t}", cat="Binary") for t in gws} for i in ids}
    s = {i: {t: pulp.LpVariable(f"s_{i}_{t}", cat="Binary") for t in gws} for i in ids}
    c = {i: {t: pulp.LpVariable(f"c_{i}_{t}", cat="Binary") for t in gws} for i in ids}
    tin = {i: {t: pulp.LpVariable(f"in_{i}_{t}", cat="Binary") for t in gws} for i in ids}
    ft = {t: pulp.LpVariable(f"ft_{t}", lowBound=1, upBound=5, cat="Integer") for t in gws}
    sv = {t: pulp.LpVariable(f"sv_{t}", lowBound=0, upBound=5, cat="Integer") for t in gws}
    hits = {t: pulp.LpVariable(f"h_{t}", lowBound=0, cat="Integer") for t in gws}

    obj = 0
    for t in gws:
        w = att[t]
        for i in ids:
            xp = float(proj.loc[i, f"xp_{t}"])
            cap_xp = float(proj.loc[i, f"cap_xp_{t}"])
            var = float(proj.loc[i, f"var_{t}"])
            eo = float(proj.loc[i, "eo"])
            obj += w * (xp * s[i][t] + bench_weight * xp * (x[i][t] - s[i][t])
                        + cap_xp * c[i][t])
            obj += -w * lambda_var * var * x[i][t]
            obj += w * lambda_eo * xp * (1.0 - eo) * x[i][t]
    # Hits cost real, not attenuated. Raised default 4 -> hit_cost so solver
    # only takes a hit when next-GW expected gain genuinely exceeds the cost.
    obj += -pulp.lpSum(hit_cost * hits[t] for t in gws)
    # Banking-roll incentive: each saved FT carries option value (pick best
    # transfer next GW). Attenuated like other per-GW terms so solver doesn't
    # game by stockpiling far-horizon saves it never spends.
    if bank_transfer_weight > 0.0:
        obj += pulp.lpSum(att[t] * bank_transfer_weight * sv[t] for t in gws)
    prob += obj

    cur_val = (proj.loc[list(current_squad_ids & set(ids))]["price"].sum()
               if current_squad_ids else 0.0)
    max_budget = cur_val + bank
    for t in gws:
        prob += pulp.lpSum(x[i][t] for i in ids) == SQUAD_SIZE
        for pos, cnt in SQUAD_COUNTS.items():
            prob += pulp.lpSum(x[i][t] for i in ids if proj.loc[i, "pos_id"] == pos) == cnt
        for tm in proj["team_id"].unique():
            prob += pulp.lpSum(x[i][t] for i in ids if proj.loc[i, "team_id"] == tm) <= MAX_PER_CLUB
        prob += pulp.lpSum(proj.loc[i, "price"] * x[i][t] for i in ids) <= max_budget

    _add_core(prob, proj, ids, s, c, lambda i, t: x[i][t], gws)

    for k, t in enumerate(gws):
        for i in ids:
            prev = (1 if (k == 0 and i in current_squad_ids)
                    else (x[i][gws[k - 1]] if k > 0 else 0))
            prob += tin[i][t] >= x[i][t] - prev
        total_in = pulp.lpSum(tin[i][t] for i in ids)
        prob += ft[t] == (free_transfers if k == 0 else 1 + sv[gws[k - 1]])
        prob += total_in == (ft[t] - sv[t]) + hits[t]
        # Per-GW hit cap. Stops solver eating multi-hit gambles in a single GW.
        prob += hits[t] <= hit_cap_per_gw

    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit))
    if pulp.LpStatus[prob.status] not in ("Optimal", "Not Solved"):
        return {"status": "infeasible"}

    squad, xi, cap, vice = _extract(proj, ids, gws, x, s, c, x_per_t=True)
    t0 = gws[0]
    return {
        "status": "ok", "gw": t0, "squad_ids": squad, "xi_ids": xi,
        "captain": cap, "vice": vice,
        "transfers_in": list(squad - current_squad_ids),
        "transfers_out": list(current_squad_ids - squad),
        "hits": int(round(float(hits[t0].varValue or 0))),
    }
