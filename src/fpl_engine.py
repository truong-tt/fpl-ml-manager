from __future__ import annotations

import itertools
import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pulp
import pymc as pm

warnings.filterwarnings("ignore")

POINTS_MAP = {1: {"goal": 6, "assist": 3, "cs": 4}, 2: {"goal": 6, "assist": 3, "cs": 4},
              3: {"goal": 5, "assist": 3, "cs": 1}, 4: {"goal": 4, "assist": 3, "cs": 0}}
SQUAD_COUNTS = {1: 2, 2: 5, 3: 5, 4: 3}
MIN_STARTERS = {1: 1, 2: 3, 3: 2, 4: 1}
POS_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
TEAM_MAP = {1: "Arsenal", 2: "Aston Villa", 3: "Burnley", 4: "Bournemouth", 5: "Brentford", 6: "Brighton",
            7: "Chelsea", 8: "Crystal Palace", 9: "Everton", 10: "Fulham", 11: "Leeds", 12: "Liverpool",
            13: "Man City", 14: "Man Utd", 15: "Newcastle", 16: "Nott'm Forest", 17: "Sunderland",
            18: "Spurs", 19: "West Ham", 20: "Wolves"}
DIXON_COLES_RHO = 0.03


class FPLEngine:
    """
    Autonomous mathematical engine for Fantasy Premier League.
    Handles match simulation, point projection, and MILP portfolio optimization.
    """

    def __init__(
            self,
            fixtures_df: pd.DataFrame,
            history_df: pd.DataFrame,
            players_df: pd.DataFrame,
            teams_df: pd.DataFrame | None = None
    ) -> None:
        """
        Initializes the engine and attempts to load the XGBoost minutes model.

        Args:
            fixtures_df (pd.DataFrame): Upcoming and historical match schedule.
            history_df (pd.DataFrame): Gameweek-level performance records for all players.
            players_df (pd.DataFrame): Static player attributes (cost, team, position).
            teams_df (pd.DataFrame | None): Static team attributes.
        """
        self.fixtures = fixtures_df
        self.history = history_df
        self.players = players_df
        self.teams = teams_df

        self.id_map: dict[int, int] = players_df.set_index('id')['team'].to_dict()
        if 'team' not in self.history.columns:
            self.history['team'] = self.history['player_id'].map(self.id_map)

        self.minutes_model = None
        model_path = Path(__file__).resolve().parent.parent / "data" / "xgboost_minutes_model.json"
        if model_path.exists():
            try:
                import xgboost as xgb
                self.minutes_model = xgb.XGBClassifier()
                self.minutes_model.load_model(model_path)
            except ImportError:
                pass
            except Exception:
                pass

    @staticmethod
    def _get_recent_data(df: pd.DataFrame, col: str, gw: int, n: int = 10, decay_rate: float = 0.15) -> pd.DataFrame:
        """Applies Exponentially Weighted Moving Average (EWMA) time-decay to past matches."""
        w = df[(df[col] >= gw - n) & (df[col] < gw)].copy()
        if w.empty: return w
        m = w[col].max()
        w['weight'] = np.exp(-decay_rate * (m - w[col]))
        return w

    @staticmethod
    def _solve_bayesian_regression(df: pd.DataFrame) -> tuple[dict[int, dict[str, float]], float, float]:
        """
        Calculates team strengths using Bayesian Poisson Regression.

        Utilizes Automatic Differentiation Variational Inference (ADVI) to approximate
        the posterior distributions of latent team strength parameters, offering a
        lightweight and fast alternative to standard MCMC stepping.

        Returns:
            tuple: (Team strengths dictionary, Home Advantage constant, Intercept constant)
        """
        if len(df) < 10:
            return {t: {'attack': 0.0, 'defense': 0.0} for t in range(1, 21)}, 0.2, 0.1

        teams = pd.unique(df[['team_h', 'team_a']].values.ravel('K'))
        team_mapping = {team: i for i, team in enumerate(teams)}
        num_teams = len(teams)

        gws = sorted(df['event'].unique())
        time_mapping = {gw: i for i, gw in enumerate(gws)}
        num_timesteps = len(gws)

        home_idx = df['team_h'].map(team_mapping).values
        away_idx = df['team_a'].map(team_mapping).values
        time_idx = df['event'].map(time_mapping).values
        home_goals = df['team_h_score'].values
        away_goals = df['team_a_score'].values

        with pm.Model() as _:
            sigma_evol = pm.HalfNormal('sigma_evol', 0.1)
            alpha = pm.GaussianRandomWalk('alpha', sigma=sigma_evol, shape=(num_teams, num_timesteps),
                                          init_dist=pm.Normal.dist(0, 0.5))
            beta = pm.GaussianRandomWalk('beta', sigma=sigma_evol, shape=(num_teams, num_timesteps),
                                         init_dist=pm.Normal.dist(0, 0.5))

            gamma = pm.Normal('home_adv', mu=0.2, sigma=0.1)
            delta = pm.Normal('intercept', mu=0, sigma=0.5)

            home_theta = pm.math.exp(
                alpha[home_idx, time_idx] + beta[away_idx, time_idx] + gamma + delta)  # type: ignore
            away_theta = pm.math.exp(alpha[away_idx, time_idx] + beta[home_idx, time_idx] + delta)  # type: ignore

            alpha_disp = pm.Exponential('alpha_disp', 1.0)
            pm.NegativeBinomial('home_obs', mu=home_theta, alpha=alpha_disp, observed=home_goals)
            pm.NegativeBinomial('away_obs', mu=away_theta, alpha=alpha_disp, observed=away_goals)

            approx = pm.fit(n=20000, progressbar=False)
            trace = approx.sample(draws=1000)

        post = getattr(trace, 'posterior', trace)
        alpha_mean = post['alpha'].mean(dim=['chain', 'draw']).values if hasattr(post, 'mean') else np.mean(
            post['alpha'], axis=0)
        beta_mean = post['beta'].mean(dim=['chain', 'draw']).values if hasattr(post, 'mean') else np.mean(post['beta'],
                                                                                                          axis=0)
        home_adv_mean = float(post['home_adv'].mean().item() if hasattr(post, 'mean') else np.mean(post['home_adv']))
        intercept_mean = float(post['intercept'].mean().item() if hasattr(post, 'mean') else np.mean(post['intercept']))

        team_strengths = {}
        for t, i in team_mapping.items():
            team_strengths[t] = {
                'attack': float(alpha_mean[i, -1] if alpha_mean.ndim > 1 else alpha_mean[i]),
                'defense': float(beta_mean[i, -1] if beta_mean.ndim > 1 else beta_mean[i])
            }

        return team_strengths, home_adv_mean, intercept_mean

    def _calculate_minute_probs(self, gw: int, n: int = 10) -> dict[int, dict[str, float]]:
        """
        Calculates expected minutes represented as a discrete 3-state probability distribution.

        Uses trained XGBoost Classifier via vectorized inference if available to predict
        the probability of Start (>= 60), Sub (> 0), or Bench (0). Falls back to EWMA decay.
        """
        past_data = self.history[self.history['round'] < gw].copy()
        if past_data.empty: return {}

        if self.minutes_model is not None:
            past_data = past_data.sort_values(['player_id', 'round'])
            last_5 = past_data.groupby('player_id').tail(5)

            features_list = []
            pids = []

            for pid, g in last_5.groupby('player_id'):
                mins = g['minutes'].values
                lag_1 = mins[-1] if len(mins) >= 1 else 0
                lag_2 = mins[-2] if len(mins) >= 2 else 0
                lag_3 = mins[-3] if len(mins) >= 3 else 0

                roll_3 = np.mean(mins[-3:]) if len(mins) > 0 else 0
                roll_5 = np.mean(mins) if len(mins) > 0 else 0
                played_last = 1 if lag_1 > 0 else 0

                features_list.append([lag_1, lag_2, lag_3, roll_3, roll_5, played_last])
                pids.append(int(cast(Any, pid)))

            if not features_list: return {}

            X = pd.DataFrame(features_list, columns=[
                'lag_1_minutes', 'lag_2_minutes', 'lag_3_minutes',
                'roll_3_avg', 'roll_5_avg', 'played_last_week'
            ])

            probs = self.minutes_model.predict_proba(X)

            res = {}
            for i, pid in enumerate(pids):
                res[pid] = {
                    'p_bench': float(probs[i][0]),
                    'p_sub': float(probs[i][1]),
                    'p_start': float(probs[i][2])
                }
            return res
        else:
            w = self._get_recent_data(self.history, 'round', gw, n)
            res = {}
            for pid, g in w.groupby('player_id'):
                p_id = int(cast(Any, pid))
                w_sum = g['weight'].sum()
                if w_sum == 0:
                    res[p_id] = {'p_start': 0.0, 'p_sub': 0.0, 'p_bench': 1.0}
                    continue

                start_w = g.loc[g['minutes'] >= 60, 'weight'].sum()
                sub_w = g.loc[(g['minutes'] > 0) & (g['minutes'] < 60), 'weight'].sum()
                bench_w = g.loc[g['minutes'] == 0, 'weight'].sum()

                res[p_id] = {
                    'p_start': start_w / w_sum,
                    'p_sub': sub_w / w_sum,
                    'p_bench': bench_w / w_sum,
                }
            return res

    def _calculate_usage_stats(self, gw: int, n: int = 10, team_strengths: dict | None = None) -> dict[
        int, dict[str, Any]]:
        """
        Allocates team-level expected metrics to individual players based on Usage Rate.

        Applies Opponent-Adjusted xG/xA mapping. It mathematically divides historic
        underlying metrics by the Bayesian defensive parameter of the opponent faced
        in that specific match, penalizing stat-padding against weak defenses.

        Args:
            gw (int): Current Gameweek.
            n (int): Size of rolling window.
            team_strengths (dict | None): Bayesian posteriors to adjust historical xG.

        Returns:
            dict: Usage rates mapped by player ID.
        """
        w = self._get_recent_data(self.history, 'round', gw, n)
        if w.empty: return {}
        std = self.history.groupby('player_id')['total_points'].std().fillna(1.0).to_dict()

        if team_strengths is not None and 'opponent_team' in w.columns:
            def get_def_multiplier(opp_id):
                opp_stats = team_strengths.get(opp_id, {'defense': 0.0})
                return np.exp(opp_stats['defense'])

            w['opp_def_multiplier'] = w['opponent_team'].apply(get_def_multiplier)
            w['adj_xg'] = w['expected_goals'] / w['opp_def_multiplier']
            w['adj_xa'] = w['expected_assists'] / w['opp_def_multiplier']
        else:
            w['adj_xg'] = w['expected_goals']
            w['adj_xa'] = w['expected_assists']

        def calc(t: pd.DataFrame):
            return pd.Series({'xg': (t['adj_xg'] * t['weight']).sum() or 0.1,
                              'xa': (t['adj_xa'] * t['weight']).sum() or 0.1,
                              'cbit': ((t['clearances_blocks_interceptions'] + t['tackles']) * t['weight']).sum() or 1})

        tt = w.groupby('team')[
            ['adj_xg', 'adj_xa', 'clearances_blocks_interceptions', 'tackles', 'weight']].apply(
            calc).to_dict('index')

        res = {}
        for pid, g in w.groupby('player_id'):
            p_id = int(cast(Any, pid))
            tid = self.id_map.get(p_id, -1)
            if g['minutes'].sum() < 45 or tid not in tt: continue
            t, wg = tt[tid], g['weight']
            res[p_id] = {'share_xg': (g['adj_xg'] * wg).sum() / t['xg'],
                         'share_xa': (g['adj_xa'] * wg).sum() / t['xa'],
                         'share_cbit': ((g['clearances_blocks_interceptions'] + g['tackles']) * wg).sum() / t['cbit'],
                         'saves_p90': g['saves'].sum() / g['minutes'].sum() * 90,
                         'risk_variance': std.get(p_id, 2.0), 'team': tid}
        return res

    def train_and_predict(self, current_gw: int, horizon: int = 5, n_recent: int = 10) -> pd.DataFrame:
        """
        Generates 3-State Expected Value (EV) projections for the optimization horizon.
        Calculates Expected Points = P(Start) * E[Pts|Start] + P(Sub) * E[Pts|Sub]
        """
        finished = self.fixtures[(self.fixtures['finished'] == True) & (self.fixtures['event'] < current_gw)]
        bayesian_strengths, home_adv, intercept = self._solve_bayesian_regression(finished)

        upcoming = self.fixtures[
            (self.fixtures['event'] >= current_gw) & (self.fixtures['event'] < current_gw + horizon)]
        match_sims = {}

        for _, row in upcoming.iterrows():
            h, a, gw = row['team_h'], row['team_a'], row['event']

            s_h = bayesian_strengths.get(h, {'attack': 0.0, 'defense': 0.0})
            s_a = bayesian_strengths.get(a, {'attack': 0.0, 'defense': 0.0})

            lambda_h = np.exp(s_h['attack'] + s_a['defense'] + home_adv + intercept)
            lambda_a = np.exp(s_a['attack'] + s_h['defense'] + intercept)

            sim_h_goals = np.random.poisson(lambda_h, 1000)
            sim_a_goals = np.random.poisson(lambda_a, 1000)

            cs_h_base = np.mean(sim_a_goals == 0)
            cs_a_base = np.mean(sim_h_goals == 0)

            match_sims[(h, gw)] = {
                'xg': lambda_h,
                'xga': lambda_a,
                'cs': min(cs_h_base * (1 + (lambda_h * lambda_a * DIXON_COLES_RHO)), 1.0),
                'sim_goals': sim_h_goals
            }
            match_sims[(a, gw)] = {
                'xg': lambda_a,
                'xga': lambda_h,
                'cs': min(cs_a_base * (1 + (lambda_h * lambda_a * DIXON_COLES_RHO)), 1.0),
                'sim_goals': sim_a_goals
            }

        usage = self._calculate_usage_stats(current_gw, n_recent, team_strengths=bayesian_strengths)
        min_probs = self._calculate_minute_probs(current_gw, 10)

        projections = []
        for _, p in self.players.iterrows():
            pid, pos = p['id'], p['element_type']
            chance = p.get('chance_of_playing_next_round', 100)
            chance = 100 if pd.isna(chance) else chance
            if p['status'] in ['s', 'n'] or chance < 25: continue

            stats = usage.get(pid, {'share_xg': 0, 'share_xa': 0, 'share_cbit': 0, 'saves_p90': 0, 'team': None})
            if stats['share_xg'] == 0 and stats['share_xa'] == 0 and pos > 1: continue

            fitness_factor = chance / 100.0
            m_prob = min_probs.get(pid, {'p_start': 0.5, 'p_sub': 0.2, 'p_bench': 0.3})
            p_start = m_prob['p_start'] * fitness_factor
            p_sub = m_prob['p_sub'] * fitness_factor

            pts = POINTS_MAP[pos]
            row = {'id': pid, 'name': p['web_name'], 'team_id': p['team'], 'pos_id': pos,
                   'price': p['now_cost'] / 10.0, 'horizon_xp': 0.0, 'next_gw_xp': 0.0,
                   'risk_variance': stats.get('risk_variance', 2.0)}

            for gw in range(current_gw, current_gw + horizon):
                row[f'xp_{gw}'] = 0.0

            for gw in range(current_gw, current_gw + horizon):
                if (p['team'], gw) not in match_sims: continue
                sim = match_sims[(p['team'], gw)]
                pxg, pxa = sim['xg'] * min(stats['share_xg'], 0.70), sim['xg'] * min(stats['share_xa'], 0.50)

                if pos in [1, 2]:
                    bps = np.mean((sim['sim_goals'] * stats['share_xg']) + (stats['share_cbit'] * 2.0) > 0.6)
                else:
                    bps = np.mean(sim['sim_goals'] * stats['share_xg'] > 0.6)

                mbps = 2.5 * bps if pos in [1, 2] else 1.5 * bps

                base_pts = (pxg * pts['goal'] + pxa * pts['assist'] +
                            (stats['saves_p90'] / 3.0 if pos == 1 else mbps) +
                            (sim['xga'] / 2.0 * -1 if pos in [1, 2] else 0))

                ev_start = p_start * (base_pts + 2 + (pts['cs'] * sim['cs']))
                ev_sub = p_sub * ((base_pts * 0.3) + 1)

                xp = ev_start + ev_sub
                row['horizon_xp'] += xp
                if gw == current_gw: row['next_gw_xp'] = xp
                row[f'xp_{gw}'] = xp

            projections.append(row)
        return pd.DataFrame(projections)

    @staticmethod
    def optimize_squad(df: pd.DataFrame, budget: float = 100.0, risk_aversion: float = 0.05,
                       stack_bonus: float = 2.5,
                       synergy_pairs: list[tuple[int, int, float]] | None = None) -> pd.DataFrame:
        """
        Solves the combinatorial knapsack problem using MILP.
        Maximizes risk-adjusted returns bounded by exact FPL rules.
        """
        if df.empty: return pd.DataFrame()
        if synergy_pairs is None: synergy_pairs = []

        prob = pulp.LpProblem("FPL_Squad", pulp.LpMaximize)
        ids = df['id'].tolist()

        x = pulp.LpVariable.dicts("p", ids, cat='Binary')
        c = pulp.LpVariable.dicts("c", ids, cat='Binary')

        teams = df['team_id'].unique()
        y = pulp.LpVariable.dicts("stack", teams, cat='Binary')

        syn_vars = pulp.LpVariable.dicts("syn", range(len(synergy_pairs)), cat='Binary')

        def g(i, col):
            return df.loc[df['id'] == i, col].values[0]

        objective = pulp.lpSum(
            [(g(i, 'horizon_xp') - (risk_aversion * g(i, 'risk_variance'))) * (x[i] + c[i]) for i in ids])
        objective += pulp.lpSum([stack_bonus * y[t] for t in teams])
        if synergy_pairs:
            objective += pulp.lpSum([synergy_pairs[idx][2] * syn_vars[idx] for idx in range(len(synergy_pairs))])

        prob += objective

        prob += pulp.lpSum([g(i, 'price') * x[i] for i in ids]) <= budget
        prob += pulp.lpSum([x[i] for i in ids]) == 15

        for pos, count in SQUAD_COUNTS.items():
            prob += pulp.lpSum([x[i] for i in ids if g(i, 'pos_id') == pos]) == count

        for t in teams:
            prob += pulp.lpSum([x[i] for i in ids if g(i, 'team_id') == t]) <= 3
            def_count = pulp.lpSum([x[i] for i in ids if g(i, 'team_id') == t and g(i, 'pos_id') in [1, 2]])
            prob += def_count >= 2 * y[t]

        for idx, (p1_id, p2_id, bonus) in enumerate(synergy_pairs):
            if p1_id in ids and p2_id in ids:
                prob += syn_vars[idx] <= x[p1_id]
                prob += syn_vars[idx] <= x[p2_id]
                prob += syn_vars[idx] >= x[p1_id] + x[p2_id] - 1
            else:
                prob += syn_vars[idx] == 0

        prob += pulp.lpSum([c[i] for i in ids]) == 1
        for i in ids: prob += c[i] <= x[i]

        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        if pulp.LpStatus[prob.status] != 'Optimal': return pd.DataFrame()

        captain = next(i for i in ids if c[i].varValue == 1)
        res = df[df['id'].isin([i for i in ids if x[i].varValue == 1])].copy()
        res['is_captain_choice'] = (res['id'] == captain).astype(int)

        return res

    @staticmethod
    def pick_team_sheet(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
        """Separates the optimized 15-man squad into Starting XI and Substitutes based on EV."""
        s = df.sort_values('next_gw_xp', ascending=False)
        xi_ids = []
        for p, m in MIN_STARTERS.items(): xi_ids.extend(s[s['pos_id'] == p].head(m)['id'].tolist())
        xi_ids.extend(s[(~s['id'].isin(xi_ids)) & (s['pos_id'] != 1)].head(11 - len(xi_ids))['id'].tolist())
        xi, bench = s[s['id'].isin(xi_ids)].sort_values('pos_id'), s[~s['id'].isin(xi_ids)].sort_values('pos_id')
        cap = s[s['is_captain_choice'] == 1]['id'].iloc[0] if 'is_captain_choice' in s and (
                s['is_captain_choice'] == 1).any() and s[s['is_captain_choice'] == 1]['id'].iloc[0] in xi_ids else \
            xi.sort_values('next_gw_xp', ascending=False).iloc[0]['id']
        vice = xi[xi['id'] != cap].sort_values('next_gw_xp', ascending=False).iloc[0]['id']
        return xi, bench, int(cap), int(vice)

    def recommend_transfers(self, squad: pd.DataFrame, bank: float, free: int = 1, gw: int | None = None,
                            proj: pd.DataFrame | None = None, risk: float = 0.05, horizon: int = 5) -> dict[str, Any]:
        """
        Evaluates optimal transfers using Receding Horizon Control (RHC).

        Transitions from a static 1D block optimization to a 2D matrix (player x time)
        to mathematically conserve and bank Free Transfers across the multi-week horizon.
        This allows the solver to evaluate the true utility of executing transfers versus
        rolling them to future gameweeks.

        Args:
            squad (pd.DataFrame): The current 15-man squad.
            bank (float): Available budget in the bank.
            free (int, optional): Number of free transfers currently available (1 to 5). Defaults to 1.
            gw (int | None, optional): The starting Gameweek for the horizon. Defaults to None.
            proj (pd.DataFrame | None, optional): Pre-calculated player projections. Defaults to None.
            risk (float, optional): Risk aversion coefficient (variance penalty). Defaults to 0.05.
            horizon (int, optional): Number of gameweeks to look ahead. Defaults to 5.

        Returns:
            dict[str, Any]: A dictionary containing the recommended 'action' (e.g., '1_TRANSFER'),
                            a list of 'transfers_made' tuples, the 'net_score' (EV), and the 'cost' (hits).
        """
        up = self.fixtures[~self.fixtures['finished']]
        gw_val = 38 if up.empty else int(up['event'].min()) if gw is None else int(gw)
        if proj is None:
            proj = self.train_and_predict(gw_val, horizon)

        current_ids = set(squad['id'].tolist())
        current_value = squad['price'].sum()
        max_budget = current_value + bank

        prob = pulp.LpProblem("FPL_RHC_Transfers", pulp.LpMaximize)
        ids = proj['id'].tolist()
        gws = list(range(gw_val, gw_val + horizon))

        x = pulp.LpVariable.dicts("x", (ids, gws), cat='Binary')
        transfer_in = pulp.LpVariable.dicts("in", (ids, gws), cat='Binary')

        ft = pulp.LpVariable.dicts("ft", gws, lowBound=1, upBound=5, cat='Integer')
        saved_ft = pulp.LpVariable.dicts("saved_ft", gws, lowBound=0, upBound=5, cat='Integer')
        hits = pulp.LpVariable.dicts("hits", gws, lowBound=0, cat='Integer')

        def g(i, col):
            return proj.loc[proj['id'] == i, col].values[0]

        obj = pulp.lpSum([
            (g(i, f'xp_{t}') - (risk * g(i, 'risk_variance') / horizon)) * x[i][t]
            for i in ids for t in gws
        ]) - pulp.lpSum([4 * hits[t] for t in gws])
        prob += obj

        for t_idx, t in enumerate(gws):
            prob += pulp.lpSum([g(i, 'price') * x[i][t] for i in ids]) <= max_budget
            prob += pulp.lpSum([x[i][t] for i in ids]) == 15

            for pos, count in SQUAD_COUNTS.items():
                prob += pulp.lpSum([x[i][t] for i in ids if g(i, 'pos_id') == pos]) == count

            for team in proj['team_id'].unique():
                prob += pulp.lpSum([x[i][t] for i in ids if g(i, 'team_id') == team]) <= 3

            for i in ids:
                if t_idx == 0:
                    was_in_squad = 1 if i in current_ids else 0
                else:
                    was_in_squad = x[i][gws[t_idx - 1]]

                prob += transfer_in[i][t] >= x[i][t] - was_in_squad

            total_transfers = pulp.lpSum([transfer_in[i][t] for i in ids])

            if t_idx == 0:
                prob += ft[t] == free
            else:
                prob += ft[t] == 1 + saved_ft[gws[t_idx - 1]]

            prob += total_transfers == ft[t] - saved_ft[t] + hits[t]

        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        if pulp.LpStatus[prob.status] != 'Optimal':
            return {'action': 'HOLD', 'transfers_made': [], 'net_score': 0, 'cost': 0}

        gw1 = gws[0]
        new_squad_ids = {i for i in ids if x[i][gw1].varValue == 1}
        players_out = current_ids - new_squad_ids
        players_in = new_squad_ids - current_ids

        cost = int(hits[gw1].varValue) * 4
        action_name = f"{len(players_in)}_TRANSFERS" if len(players_in) > 0 else "HOLD"

        moves = [(g(o, 'name'), g(in_id, 'name')) for o, in_id in zip(players_out, players_in)]

        return {
            'action': action_name,
            'transfers_made': moves,
            'net_score': round(pulp.value(prob.objective), 2),
            'cost': cost
        }

    def get_transfer_recommendation_str(self, rec: dict[str, Any]) -> str:
        """Formats the recommended transfer actions as a string."""
        out = "\nTRANSFER SUMMARY\n" + self._to_ascii_table(pd.DataFrame(
            [{'Action': rec['action'], 'Cost': f"-{rec['cost']}", 'Net RHC EV': float(rec['net_score'])}]))
        if rec['action'] != 'HOLD':
            out += "\nTRANSFERS\n" + self._to_ascii_table(
                pd.DataFrame([{'Sell': s, 'Buy': b} for s, b in rec['transfers_made']]))
        return out

    @staticmethod
    def _to_ascii_table(df: pd.DataFrame | None) -> str:
        """Utility function to render DataFrames as CLI-friendly ASCII tables."""
        if df is None or df.empty: return "(empty)"
        df = df.fillna('')
        c, r = [str(col) for col in df.columns], [[str(v) for v in row] for _, row in df.iterrows()]
        w = [max(len(col), max((len(row[i]) for row in r), default=0)) for i, col in enumerate(c)]
        b = '+' + '+'.join('-' * (width + 2) for width in w) + '+'

        def fmt(v): return '| ' + ' | '.join(str(val).ljust(width) for val, width in zip(v, w)) + ' |'

        return '\n'.join([b, fmt(c), b] + [fmt(row) for row in r] + [b])

    def format_squad(self, df: pd.DataFrame) -> pd.DataFrame:
        """Formats squad output for display."""
        out = df.copy()
        out['Team'], out['Pos'] = out['team_id'].map(TEAM_MAP).fillna(out['team_id']), out['pos_id'].map(POS_MAP)
        out['XP(1)'], out['XP(5)'], out['Price'] = out['next_gw_xp'].round(1), out['horizon_xp'].round(1), out[
            'price'].round(1)
        out['Role'] = out['is_captain_choice'].map({1: '(C)'}).fillna('') if 'is_captain_choice' in out.columns else ''
        return out[['name', 'Team', 'Pos', 'Price', 'XP(1)', 'XP(5)', 'Role']].rename(columns={'name': 'Name'})

    def get_squad_str(self, xi: pd.DataFrame, bench: pd.DataFrame, cap: int, vice: int) -> str:
        """Formats the Starting XI and Bench as a string."""
        out = ""
        for title, d in [("STARTING XI", xi), ("SUBSTITUTES", bench)]:
            out += f"\n{title}\n"
            df = self.format_squad(d).copy()
            if title == "STARTING XI":
                df.loc[xi['id'] == cap, 'Role'] = 'C'
                df.loc[xi['id'] == vice, 'Role'] = 'VC'
            out += self._to_ascii_table(df) + "\n"
        return out