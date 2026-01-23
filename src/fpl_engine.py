import pandas as pd
import numpy as np
import pulp
from scipy.optimize import minimize

# FPL scoring (subset used by the xP model)
POINTS_MAP = {
    1: {"goal": 6, "assist": 3, "cs": 4},  # GK
    2: {"goal": 6, "assist": 3, "cs": 4},  # DEF
    3: {"goal": 5, "assist": 3, "cs": 1},  # MID
    4: {"goal": 4, "assist": 3, "cs": 0},  # FWD
}

# 15-man squad constraints
SQUAD_COUNTS = {1: 2, 2: 5, 3: 5, 4: 3}

# Starting XI formation (fixed)
STARTING_COUNTS = {1: 1, 2: 4, 3: 4, 4: 2}


class FPLEngine:
    def __init__(self, fixtures_path, history_path, players_path):
        # Load inputs
        self.fixtures = pd.read_csv(fixtures_path)
        self.history = pd.read_csv(history_path)
        self.players = pd.read_csv(players_path)

        # Player -> team for history rows
        self.id_map = self.players.set_index("id")["team"].to_dict()
        self.history["team"] = self.history["player_id"].map(self.id_map)

    def _solve_poisson_regression(self, df):
        """Fit team attack/defense via Poisson MLE."""
        teams = sorted(set(df["team_h"]) | set(df["team_a"]))
        n = len(teams)

        def loss(params):
            home_adv, intercept = params[0], params[1]
            alphas = dict(zip(teams, params[2 : 2 + n]))
            betas = dict(zip(teams, params[2 + n :]))

            mu_h = np.exp(intercept + home_adv + df["team_h"].map(alphas) + df["team_a"].map(betas))
            mu_a = np.exp(intercept + df["team_a"].map(alphas) + df["team_h"].map(betas))

            ll = (df["team_h_score"] * np.log(mu_h) - mu_h) + (df["team_a_score"] * np.log(mu_a) - mu_a)
            return -np.sum(ll)

        init = np.concatenate([[0.2, 0.0], np.zeros(n * 2)])
        res = minimize(loss, init, method="L-BFGS-B", options={"maxiter": 100})

        p = res.x
        return p[0], p[1], dict(zip(teams, p[2 : 2 + n])), dict(zip(teams, p[2 + n :]))

    def train_and_predict(self, current_gw, horizon=3):
        """Train team model, project player xP (horizon + next GW)."""
        train_df = self.fixtures[(self.fixtures["event"] < current_gw) & self.fixtures["finished"]].copy()

        # Fallback if too little data
        if len(train_df) < 20:
            home_adv, intercept, alphas, betas = 0.2, 0.0, {}, {}
        else:
            home_adv, intercept, alphas, betas = self._solve_poisson_regression(train_df)

        # Fixture-level goal/cs expectations
        horizon_gws = range(current_gw, current_gw + horizon)
        future = self.fixtures[self.fixtures["event"].isin(horizon_gws)]

        fix_preds = []
        for _, row in future.iterrows():
            h, a = row["team_h"], row["team_a"]
            lambda_h = np.exp(intercept + home_adv + alphas.get(h, 0) + betas.get(a, 0))
            lambda_a = np.exp(intercept + alphas.get(a, 0) + betas.get(h, 0))

            fix_preds.append({"event": row["event"], "team": h, "goals": lambda_h, "cs": np.exp(-lambda_a)})
            fix_preds.append({"event": row["event"], "team": a, "goals": lambda_a, "cs": np.exp(-lambda_h)})

        df_fix = pd.DataFrame(fix_preds)
        if df_fix.empty:
            return pd.DataFrame()

        # Player projections from per-90 + minutes + availability
        past = self.history[self.history["round"] < current_gw]
        projections = []

        for _, p in self.players.iterrows():
            chance = p["chance_of_playing_next_round"] or 100.0
            if p["status"] in ["s", "n"] or chance < 25:
                continue

            p_hist = past[past["player_id"] == p["id"]]
            active = p_hist[p_hist["minutes"] > 0]
            if len(active) < 3:
                continue

            role_mins = active.tail(5)["minutes"].mean()
            if role_mins < 30:
                continue

            prob_play = (chance / 100.0) * (0.8 if chance == 75 else 1.0)
            mins_sum = p_hist["minutes"].sum()
            g_90 = (p_hist["goals_scored"].sum() / mins_sum) * 90 if mins_sum else 0
            a_90 = (p_hist["assists"].sum() / mins_sum) * 90 if mins_sum else 0

            team_fix = df_fix[df_fix["team"] == p["team"]]
            total_xp, next_gw_xp = 0, 0
            pos = p["element_type"]
            pts = POINTS_MAP[pos]

            for _, f in team_fix.iterrows():
                play_factor = role_mins / 90.0
                xp = (
                    f["goals"]
                    * play_factor
                    * (g_90 * pts["goal"] + a_90 * pts["assist"])
                    / 1.3
                    * prob_play
                )

                if pos in [1, 2] and role_mins >= 60:
                    xp += f["cs"] * pts["cs"] * prob_play

                xp += (2 if role_mins >= 60 else 1) * prob_play

                total_xp += xp
                if f["event"] == current_gw:
                    next_gw_xp = xp

            projections.append(
                {
                    "id": p["id"],
                    "name": p["web_name"],
                    "team": p["team"],
                    "pos": pos,
                    "price": p["now_cost"] / 10.0,
                    "horizon_xp": total_xp,
                    "next_gw_xp": next_gw_xp,
                }
            )

        return pd.DataFrame(projections)

    def optimize_squad(self, df, budget=100.0):
        """Max-xP 15-man squad under budget/position/team caps."""
        prob = pulp.LpProblem("FPL", pulp.LpMaximize)
        ids = df["id"].tolist()
        x = pulp.LpVariable.dicts("p", ids, cat="Binary")

        prob += pulp.lpSum([df.loc[df["id"] == i, "horizon_xp"].values[0] * x[i] for i in ids])
        prob += pulp.lpSum([df.loc[df["id"] == i, "price"].values[0] * x[i] for i in ids]) <= budget
        prob += pulp.lpSum(x.values()) == 15

        for pos, count in SQUAD_COUNTS.items():
            prob += pulp.lpSum([x[i] for i in ids if df.loc[df["id"] == i, "pos"].values[0] == pos]) == count

        for t in df["team"].unique():
            prob += pulp.lpSum([x[i] for i in ids if df.loc[df["id"] == i, "team"].values[0] == t]) <= 3

        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        if pulp.LpStatus[prob.status] != "Optimal":
            return pd.DataFrame()

        selected = [i for i in ids if x[i].varValue == 1]
        return df[df["id"].isin(selected)].copy()

    def pick_team_sheet(self, squad_df):
        """Pick XI + bench, then set captain/vice by next-GW xP."""
        squad_df = squad_df.sort_values("next_gw_xp", ascending=False)

        starters = pd.concat(
            [squad_df[squad_df["pos"] == pos].head(count) for pos, count in STARTING_COUNTS.items()]
        )

        bench = squad_df[~squad_df["id"].isin(starters["id"])].sort_values("next_gw_xp", ascending=False)
        starters = starters.sort_values("next_gw_xp", ascending=False)

        return starters, bench, starters.iloc[0]["id"], starters.iloc[1]["id"]