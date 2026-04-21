from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pulp

warnings.filterwarnings("ignore")

POINTS_MAP = {1: {"goal": 10, "assist": 3, "cs": 4}, 2: {"goal": 6, "assist": 3, "cs": 4},
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
    """Autonomous engine for FPL match simulation and optimization."""

    def __init__(
            self,
            fixtures_df: pd.DataFrame,
            history_df: pd.DataFrame,
            players_df: pd.DataFrame,
            teams_df: pd.DataFrame | None = None
    ) -> None:
        """Initializes engine and loads XGBoost models.

        Args:
            fixtures_df: Upcoming and historical match schedule.
            history_df: Gameweek-level performance records.
            players_df: Static player attributes.
            teams_df: Static team attributes.
        """
        self.fixtures = fixtures_df
        self.history = history_df
        self.players = players_df
        self.teams = teams_df

        self.id_map: dict[int, int] = players_df.set_index('id')['team'].to_dict()
        if 'team' not in self.history.columns:
            self.history['team'] = self.history['player_id'].map(self.id_map)

        self.minutes_model = None
        self.match_model_h = None
        self.match_model_a = None

        data_dir = Path(__file__).resolve().parent.parent / "data"
        try:
            import xgboost as xgb
            model_path = data_dir / "xgboost_minutes_model.json"
            if model_path.exists():
                self.minutes_model = xgb.XGBClassifier()
                self.minutes_model.load_model(model_path)

            model_h_path = data_dir / "xgb_home_goals.json"
            if model_h_path.exists():
                self.match_model_h = xgb.Booster()
                self.match_model_h.load_model(model_h_path)

            model_a_path = data_dir / "xgb_away_goals.json"
            if model_a_path.exists():
                self.match_model_a = xgb.Booster()
                self.match_model_a.load_model(model_a_path)
        except ImportError:
            pass

    @staticmethod
    def _get_recent_data(df: pd.DataFrame, col: str, gw: int, n: int = 10, decay_rate: float = 0.15) -> pd.DataFrame:
        """Applies EWMA time-decay to past matches.

        Args:
            df: Input dataframe.
            col: Column representing gameweek/time.
            gw: Current gameweek.
            n: Window size.
            decay_rate: Exponential decay factor.

        Returns:
            Dataframe with calculated weights.
        """
        w = df[(df[col] >= gw - n) & (df[col] < gw)].copy()
        if w.empty: return w
        m = w[col].max()
        w['weight'] = np.exp(-decay_rate * (m - w[col]))
        return w

    def _calculate_minute_probs(self, gw: int, n: int = 10) -> dict[int, dict[str, float]]:
        """Calculates expected minutes probabilities.

        Args:
            gw: Current gameweek.
            n: Window size.

        Returns:
            Mapping of player IDs to minute probabilities.
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
        """Allocates team-level expected metrics to individual players.

        Args:
            gw: Current Gameweek.
            n: Window size.
            team_strengths: Opponent defensive and attacking strengths.

        Returns:
            Usage rates mapped by player ID.
        """
        w = self._get_recent_data(self.history, 'round', gw, n)
        if w.empty: return {}
        std = self.history.groupby('player_id')['total_points'].std().fillna(1.0).to_dict()

        if team_strengths is not None and 'opponent_team' in w.columns:
            def get_def_multiplier(opp_id):
                return np.exp(team_strengths.get(opp_id, {'defense': 0.0})['defense'])

            def get_atk_multiplier(opp_id):
                return np.exp(team_strengths.get(opp_id, {'attack': 0.0})['attack'])

            w['opp_def_multiplier'] = w['opponent_team'].apply(get_def_multiplier)
            w['opp_atk_multiplier'] = w['opponent_team'].apply(get_atk_multiplier)

            w['adj_xg'] = w['expected_goals'] / w['opp_def_multiplier']
            w['adj_xa'] = w['expected_assists'] / w['opp_def_multiplier']

            w['adj_cbit'] = (w['clearances_blocks_interceptions'] + w['tackles']) / w['opp_atk_multiplier']
            w['adj_cbirt'] = (w['clearances_blocks_interceptions'] + w['tackles'] + w.get('recoveries', 0)) / w[
                'opp_atk_multiplier']
        else:
            w['adj_xg'] = w['expected_goals']
            w['adj_xa'] = w['expected_assists']
            w['adj_cbit'] = w['clearances_blocks_interceptions'] + w['tackles']
            w['adj_cbirt'] = w['adj_cbit'] + w.get('recoveries', 0)

        def calc(t: pd.DataFrame):
            return pd.Series({'xg': (t['adj_xg'] * t['weight']).sum() or 0.1,
                              'xa': (t['adj_xa'] * t['weight']).sum() or 0.1})

        tt = w.groupby('team').apply(calc).to_dict('index')

        res = {}
        for pid, g in w.groupby('player_id'):
            p_id = int(cast(Any, pid))
            tid = self.id_map.get(p_id, -1)
            if g['minutes'].sum() < 45 or tid not in tt: continue
            t, wg, mins = tt[tid], g['weight'], g['minutes'].sum()

            neg = g.get('yellow_cards', 0) + g.get('red_cards', 0) * 3 + g.get('penalties_missed', 0) * 2 + g.get(
                'own_goals', 0) * 2

            res[p_id] = {'share_xg': (g['adj_xg'] * wg).sum() / t['xg'],
                         'share_xa': (g['adj_xa'] * wg).sum() / t['xa'],
                         'cbit_p90': (g['adj_cbit'] * wg).sum() / mins * 90,
                         'cbirt_p90': (g['adj_cbirt'] * wg).sum() / mins * 90,
                         'neg_p90': (neg * wg).sum() / mins * 90,
                         'saves_p90': g['saves'].sum() / mins * 90,
                         'risk_variance': std.get(p_id, 2.0), 'team': tid}
        return res

    def train_and_predict(self, current_gw: int, horizon: int = 5, n_recent: int = 10) -> pd.DataFrame:
        """Generates projections for optimization horizon.

        Args:
            current_gw: Target gameweek.
            horizon: Weeks to predict.
            n_recent: Lookback window.

        Returns:
            Dataframe of player point projections.
        """
        from src.features import build_match_features
        import xgboost as xgb

        upcoming = self.fixtures[(self.fixtures['event'] >= current_gw) & (self.fixtures['event'] < current_gw + horizon)]
        match_features = build_match_features(upcoming, self.history)

        match_sims = {}
        for _, row in match_features.iterrows():
            h, a, gw = row['team_h'], row['team_a'], row['event']

            if self.match_model_h and self.match_model_a:
                X = xgb.DMatrix(pd.DataFrame([row[['h_xg', 'h_xga', 'a_xg', 'a_xga', 'strength_diff']]]))
                lambda_h = self.match_model_h.predict(X)[0]
                lambda_a = self.match_model_a.predict(X)[0]
            else:
                lambda_h, lambda_a = 1.5, 1.0

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

        usage = self._calculate_usage_stats(current_gw, n_recent)
        min_probs = self._calculate_minute_probs(current_gw, 10)

        projections = []
        for _, p in self.players.iterrows():
            pid, pos = p['id'], p['element_type']
            chance = p.get('chance_of_playing_next_round', 100)
            chance = 100 if pd.isna(chance) else chance
            if p['status'] in ['s', 'n'] or chance < 25: continue

            stats = usage.get(pid, {'share_xg': 0, 'share_xa': 0, 'team': None})
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

                bps = np.mean(sim['sim_goals'] * stats['share_xg'] > 0.6)
                mbps = 2.5 * bps if pos in [1, 2] else 1.5 * bps

                def_action_pts = 0
                if pos == 2:
                    def_action_pts = np.mean(np.random.poisson(stats.get('cbit_p90', 0), 1000) >= 10) * 2
                elif pos in [3, 4]:
                    def_action_pts = np.mean(np.random.poisson(stats.get('cbirt_p90', 0), 1000) >= 12) * 2

                neg_pts = stats.get('neg_p90', 0)

                base_pts = (pxg * pts['goal'] + pxa * pts['assist'] +
                            (stats.get('saves_p90', 0) / 3.0 if pos == 1 else mbps) +
                            def_action_pts - neg_pts +
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
        """Runs MILP optimization to select squad.

        Args:
            df: Projections dataframe.
            budget: Constraints.
            risk_aversion: Penalty.
            stack_bonus: Multiplier.
            synergy_pairs: Pairings.

        Returns:
            Optimized dataframe.
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
        """Splits squad into XI and subs."""
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
        """Evaluates optimal transfers via RHC."""
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
        """Formats transfer actions as string."""
        out = "\nTRANSFER SUMMARY\n" + self._to_ascii_table(pd.DataFrame(
            [{'Action': rec['action'], 'Cost': f"-{rec['cost']}", 'Net RHC EV': float(rec['net_score'])}]))
        if rec['action'] != 'HOLD':
            out += "\nTRANSFERS\n" + self._to_ascii_table(
                pd.DataFrame([{'Sell': s, 'Buy': b} for s, b in rec['transfers_made']]))
        return out

    @staticmethod
    def _to_ascii_table(df: pd.DataFrame | None) -> str:
        """Utility to render DataFrames as tables."""
        if df is None or df.empty: return "(empty)"
        df = df.fillna('')
        c, r = [str(col) for col in df.columns], [[str(v) for v in row] for _, row in df.iterrows()]
        w = [max(len(col), max((len(row[i]) for row in r), default=0)) for i, col in enumerate(c)]
        b = '+' + '+'.join('-' * (width + 2) for width in w) + '+'

        def fmt(v): return '| ' + ' | '.join(str(val).ljust(width) for val, width in zip(v, w)) + ' |'

        return '\n'.join([b, fmt(c), b] + [fmt(row) for row in r] + [b])

    def format_squad(self, df: pd.DataFrame) -> pd.DataFrame:
        """Formats squad output."""
        out = df.copy()
        out['Team'], out['Pos'] = out['team_id'].map(TEAM_MAP).fillna(out['team_id']), out['pos_id'].map(POS_MAP)
        out['XP(1)'], out['XP(5)'], out['Price'] = out['next_gw_xp'].round(1), out['horizon_xp'].round(1), out[
            'price'].round(1)
        out['Role'] = out['is_captain_choice'].map({1: '(C)'}).fillna('') if 'is_captain_choice' in out.columns else ''
        return out[['name', 'Team', 'Pos', 'Price', 'XP(1)', 'XP(5)', 'Role']].rename(columns={'name': 'Name'})

    def get_squad_str(self, xi: pd.DataFrame, bench: pd.DataFrame, cap: int, vice: int) -> str:
        """Formats XI and Bench as string."""
        out = ""
        for title, d in [("STARTING XI", xi), ("SUBSTITUTES", bench)]:
            out += f"\n{title}\n"
            df = self.format_squad(d).copy()
            if title == "STARTING XI":
                df.loc[xi['id'] == cap, 'Role'] = 'C'
                df.loc[xi['id'] == vice, 'Role'] = 'VC'
            out += self._to_ascii_table(df) + "\n"
        return out