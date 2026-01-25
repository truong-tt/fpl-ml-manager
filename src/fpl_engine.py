from __future__ import annotations

import itertools
from typing import Any, Hashable

import numpy as np
import pandas as pd
import pulp
from scipy.optimize import minimize

# --- CONSTANTS ---
POINTS_MAP: dict[int, dict[str, int]] = {
    1: {'goal': 6, 'assist': 3, 'cs': 4},  # GK
    2: {'goal': 6, 'assist': 3, 'cs': 4},  # DEF
    3: {'goal': 5, 'assist': 3, 'cs': 1},  # MID
    4: {'goal': 4, 'assist': 3, 'cs': 0}   # FWD
}
SQUAD_COUNTS: dict[int, int] = {1: 2, 2: 5, 3: 5, 4: 3}
MIN_STARTERS: dict[int, int] = {1: 1, 2: 3, 3: 2, 4: 1}
POS_MAP: dict[int, str] = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
TEAM_MAP: dict[int, str] = {
    1: "Arsenal", 2: "Aston Villa", 3: "Burnley", 4: "Bournemouth",
    5: "Brentford", 6: "Brighton", 7: "Chelsea", 8: "Crystal Palace",
    9: "Everton", 10: "Fulham", 11: "Leeds", 12: "Liverpool",
    13: "Man City", 14: "Man Utd", 15: "Newcastle", 16: "Nott'm Forest",
    17: "Sunderland", 18: "Spurs", 19: "West Ham", 20: "Wolves"
}
RECENCY_WEIGHTS: list[float] = [1.0, 1.0, 0.8, 0.6, 0.4, 0.3, 0.2]
DEFAULT_STRENGTH: dict[str, float] = {'strength': 0.6, 'strength_home': 1.15, 'strength_away': 1.15}
DEFAULT_USAGE_STATS: dict[str, float | None] = {'share_xg': 0, 'share_xa': 0, 'share_cbit': 0, 'saves_p90': 0, 'team': None}

class FPLEngine:
    def __init__(
        self,
        fixtures_df: pd.DataFrame,
        history_df: pd.DataFrame,
        players_df: pd.DataFrame,
        teams_df: pd.DataFrame | None = None
    ) -> None:
        self.fixtures = fixtures_df
        self.history = history_df
        self.players = players_df
        self.teams = teams_df
        self.id_map: dict[int, int] = players_df.set_index('id')['team'].to_dict()
        
        if 'team' not in self.history.columns:
            self.history['team'] = self.history['player_id'].map(self.id_map)
        
        # Team strength lookups (normalized for regression stability)
        self.team_stats: dict[int, dict[str, float]] = {}
        if teams_df is not None:
            for _, t in teams_df.iterrows():
                self.team_stats[t['team_id']] = {
                    'strength': t['strength'] / 5.0,
                    'strength_home': t['strength_overall_home'] / 1000.0,
                    'strength_away': t['strength_overall_away'] / 1000.0
                }

    def _get_team_stat(self, team_id: int, key: str) -> float:
        return self.team_stats.get(team_id, DEFAULT_STRENGTH).get(key, DEFAULT_STRENGTH[key])

    def _get_recent_data(self, df: pd.DataFrame, gw_col: str, current_gw: int, n_recent: int = 7) -> pd.DataFrame:
        """Filter to recent gameweeks and add recency weights."""
        window = df[(df[gw_col] >= current_gw - n_recent) & (df[gw_col] < current_gw)].copy()
        if window.empty:
            return window
        max_gw = window[gw_col].max()
        window['weight'] = window[gw_col].apply(
            lambda gw: RECENCY_WEIGHTS[int(max_gw - gw)] if max_gw - gw < len(RECENCY_WEIGHTS) else 0.1
        )
        return window

    def _solve_weighted_regression(self, df: pd.DataFrame, n_recent: int = 7) -> np.ndarray:
        """Weighted Poisson regression on recent matches."""
        gws = sorted(df['event'].unique())[-n_recent:]
        df = df[df['event'].isin(gws)].copy()
        
        if len(df) < 10:
            return np.array([0.1, 0.2, 0.5, 0.5])
        
        # Add time-decay weights
        max_gw = max(gws)
        df['weight'] = df['event'].apply(
            lambda gw: RECENCY_WEIGHTS[max_gw - gw] if max_gw - gw < len(RECENCY_WEIGHTS) else 0.1
        )

        # Prepare regression data
        h_str = np.asarray(df['team_h'].apply(lambda x: self._get_team_stat(x, 'strength_home')).values)
        a_str = np.asarray(df['team_a'].apply(lambda x: self._get_team_stat(x, 'strength_away')).values)
        h_ovr = np.asarray(df['team_h'].apply(lambda x: self._get_team_stat(x, 'strength')).values)
        a_ovr = np.asarray(df['team_a'].apply(lambda x: self._get_team_stat(x, 'strength')).values)
        y_h, y_a, w = np.asarray(df['team_h_score'].values), np.asarray(df['team_a_score'].values), np.asarray(df['weight'].values)

        def loss(params):
            intercept, home_adv, c_own, c_opp = params
            mu_h = np.exp(intercept + home_adv + c_own * (h_str + h_ovr) - c_opp * a_str)
            mu_a = np.exp(intercept + c_own * (a_str + a_ovr) - c_opp * h_str)
            ll = (y_h * np.log(mu_h + 1e-9) - mu_h + y_a * np.log(mu_a + 1e-9) - mu_a) * w
            return -ll.sum()

        res = minimize(loss, [0.1, 0.2, 0.5, 0.5], method='L-BFGS-B',
                      bounds=[(-3, 3), (0, 1), (0, 3), (0, 3)])
        return np.asarray(res.x)

    def _calculate_nailedness(self, current_gw: int, n_recent: int = 7) -> dict[int, float]:
        """Minute security from recent matches."""
        window = self._get_recent_data(self.history, 'round', current_gw, n_recent)
        if window.empty:
            return {}
        
        stats = window.groupby('player_id')['minutes'].agg(['sum', 'count'])
        team_max = window.groupby('team')['minutes'].sum().to_dict()
        
        nailed: dict[int, float] = {}
        for pid, row in stats.iterrows():
            player_id = int(pid)  # type: ignore[arg-type]
            if row['sum'] == 0:
                nailed[player_id] = 0.0
                continue
            tid = self.id_map.get(player_id)
            share = row['sum'] / max(1, team_max.get(tid, 90 * n_recent))
            nailed[player_id] = 1.05 if share >= 0.90 else 1.0 if share >= 0.75 else 0.85 if share >= 0.50 else 0.5
        return nailed

    def _calculate_usage_stats(self, current_gw: int, n_recent: int = 7) -> dict[int, dict[str, Any]]:
        """Player xG/xA share with recency weights."""
        window = self._get_recent_data(self.history, 'round', current_gw, n_recent)
        if window.empty:
            return {}
        
        # Team totals (weighted)
        def calc_team_totals(t: pd.DataFrame) -> pd.Series:
            return pd.Series({
                'xg': (t['expected_goals'] * t['weight']).sum() or 0.1,
                'xa': (t['expected_assists'] * t['weight']).sum() or 0.1,
                'cbit': ((t['clearances_blocks_interceptions'] + t['tackles']) * t['weight']).sum() or 1
            })
        
        team_totals_df = window.groupby('team')[['expected_goals', 'expected_assists', 'clearances_blocks_interceptions', 'tackles', 'weight']].apply(calc_team_totals)
        team_totals: dict[Hashable, dict[Hashable, Any]] = team_totals_df.to_dict('index')

        metrics = {}
        for pid, grp in window.groupby('player_id'):
            if grp['minutes'].sum() < 45:
                continue
            player_id = int(pid) if not isinstance(pid, int) else pid  # type: ignore[arg-type]
            tid = self.id_map.get(player_id)
            if tid not in team_totals:
                continue
            
            tt = team_totals[tid]
            w = grp['weight']
            metrics[pid] = {
                'share_xg': (grp['expected_goals'] * w).sum() / tt['xg'],
                'share_xa': (grp['expected_assists'] * w).sum() / tt['xa'],
                'share_cbit': ((grp['clearances_blocks_interceptions'] + grp['tackles']) * w).sum() / tt['cbit'],
                'saves_p90': grp['saves'].sum() / grp['minutes'].sum() * 90,
                'team': tid
            }
        return metrics

    def train_and_predict(self, current_gw: int, horizon: int = 5, n_recent: int = 7) -> pd.DataFrame:
        """Train on recent data, predict next gameweeks."""
        finished = self.fixtures[(self.fixtures['finished'] == True) & (self.fixtures['event'] < current_gw)]
        params = self._solve_weighted_regression(finished, n_recent) if len(finished) >= 10 else np.array([0.1, 0.2, 0.5, 0.5])
        intercept, home_adv, c_own, c_opp = params

        # Simulate upcoming fixtures
        upcoming = self.fixtures[(self.fixtures['event'] >= current_gw) & (self.fixtures['event'] < current_gw + horizon)]
        match_sims = {}
        for _, row in upcoming.iterrows():
            h, a, gw = row['team_h'], row['team_a'], row['event']
            s_h = self.team_stats.get(h, DEFAULT_STRENGTH)
            s_a = self.team_stats.get(a, DEFAULT_STRENGTH)
            
            lambda_h = np.exp(intercept + home_adv + c_own * (s_h['strength_home'] + s_h['strength']) - c_opp * s_a['strength_away'])
            lambda_a = np.exp(intercept + c_own * (s_a['strength_away'] + s_a['strength']) - c_opp * s_h['strength_home'])
            
            match_sims[(h, gw)] = {'xg': lambda_h, 'xga': lambda_a, 'cs': np.exp(-lambda_a)}
            match_sims[(a, gw)] = {'xg': lambda_a, 'xga': lambda_h, 'cs': np.exp(-lambda_h)}

        # Player projections
        usage = self._calculate_usage_stats(current_gw, n_recent)
        nailed = self._calculate_nailedness(current_gw, n_recent)
        
        projections = []
        for _, p in self.players.iterrows():
            pid, pos = p['id'], p['element_type']
            chance = p.get('chance_of_playing_next_round', 100)
            chance = 100 if pd.isna(chance) else chance
            
            if p['status'] in ['s', 'n'] or chance < 25:
                continue
            
            stats = usage.get(pid, DEFAULT_USAGE_STATS.copy())
            if stats['share_xg'] == 0 and stats['share_xa'] == 0 and pos > 1:
                continue
            
            play_prob = (chance / 100.0) * nailed.get(pid, 0.5)
            pts = POINTS_MAP[pos]
            row = {'id': pid, 'name': p['web_name'], 'team_id': p['team'],
                   'pos_id': pos, 'price': p['now_cost'] / 10.0, 'horizon_xp': 0, 'next_gw_xp': 0}
            
            for gw in range(current_gw, current_gw + horizon):
                if (p['team'], gw) not in match_sims:
                    continue
                sim = match_sims[(p['team'], gw)]
                
                # Fixed: Use sim['xg'] for goals (player's team xG) and sim['xg'] for assists 
                # (assists come from team's attacking output, not opponent's)
                xp = (sim['xg'] * min(stats['share_xg'], 0.70) * pts['goal'] +
                      sim['xg'] * min(stats['share_xa'], 0.50) * pts['assist'] +
                      pts['cs'] * sim['cs'] +
                      (sim['xga'] / 2.0 * -1 if pos in [1, 2] else 0) +
                      (stats['saves_p90'] / 3.0 if pos == 1 else 2 if sim['xg'] * stats['share_xg'] > 0.4 else 0) + 2) * play_prob
                
                row['horizon_xp'] += xp
                if gw == current_gw:
                    row['next_gw_xp'] = xp
            
            projections.append(row)
        
        return pd.DataFrame(projections)

    # --- OPTIMIZATION METHODS ---
    def optimize_squad(self, df: pd.DataFrame, budget: float = 100.0) -> pd.DataFrame:
        """Linear optimization for squad selection."""
        if df.empty:
            return pd.DataFrame()
        
        prob = pulp.LpProblem("FPL_Squad", pulp.LpMaximize)
        ids = df['id'].tolist()
        x = pulp.LpVariable.dicts("player", ids, cat='Binary')
        c = pulp.LpVariable.dicts("captain", ids, cat='Binary')
        
        # Helper to get player attribute
        def get(i: int, col: str) -> Any:
            return df.loc[df['id'] == i, col].values[0]
        
        prob += pulp.lpSum([get(i, 'horizon_xp') * (x[i] + c[i]) for i in ids])
        prob += pulp.lpSum([get(i, 'price') * x[i] for i in ids]) <= budget
        prob += pulp.lpSum([x[i] for i in ids]) == 15
        
        for pos, count in SQUAD_COUNTS.items():
            prob += pulp.lpSum([x[i] for i in ids if get(i, 'pos_id') == pos]) == count
        for t in df['team_id'].unique():
            prob += pulp.lpSum([x[i] for i in ids if get(i, 'team_id') == t]) <= 3
        
        prob += pulp.lpSum([c[i] for i in ids]) == 1
        for i in ids:
            prob += c[i] <= x[i]

        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[prob.status] != 'Optimal':
            return pd.DataFrame()
        
        selected = [i for i in ids if x[i].varValue == 1]
        captain = next(i for i in ids if c[i].varValue == 1)
        result = df[df['id'].isin(selected)].copy()
        result['is_captain_choice'] = (result['id'] == captain).astype(int)
        return result

    def pick_team_sheet(self, squad_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
        """Pick best starting XI from 15 players."""
        squad = squad_df.sort_values('next_gw_xp', ascending=False)
        
        starters: list[int] = []
        for pos, min_count in MIN_STARTERS.items():
            starters.extend(squad[squad['pos_id'] == pos].head(min_count)['id'].tolist())
        
        # Add 4 best remaining outfielders
        remaining = squad[(~squad['id'].isin(starters)) & (squad['pos_id'] != 1)]
        starters.extend(remaining.head(4)['id'].tolist())
        
        xi = squad[squad['id'].isin(starters)].sort_values('pos_id')
        bench = squad[~squad['id'].isin(starters)].sort_values('pos_id')
        
        # Captain selection - Fixed: use 'is_captain_choice' column safely
        cap_choice_col = squad['is_captain_choice'] if 'is_captain_choice' in squad.columns else pd.Series(0, index=squad.index)
        solver_cap = squad[cap_choice_col == 1]
        if not solver_cap.empty and solver_cap.iloc[0]['id'] in xi['id'].values:
            cap_id = solver_cap.iloc[0]['id']
        else:
            cap_id = xi.sort_values('next_gw_xp', ascending=False).iloc[0]['id']
        
        vice_id = xi[xi['id'] != cap_id].sort_values('next_gw_xp', ascending=False).iloc[0]['id']
        return xi, bench, cap_id, vice_id

    def recommend_transfers(
        self,
        current_squad: pd.DataFrame,
        bank: float,
        free_transfers: int = 1,
        current_gw: int | None = None
    ) -> dict[str, Any]:
        """Beam search for transfer recommendations."""
        if current_gw is None:
            upcoming = self.fixtures[~self.fixtures['finished']]
            current_gw = 38 if upcoming.empty else int(upcoming['event'].min())
        
        projections = self.train_and_predict(current_gw, horizon=5)
        current_ids = set(current_squad['id'])
        pool = projections[~projections['id'].isin(current_ids)].nlargest(50, 'horizon_xp')
        sell_cands = current_squad.nsmallest(5, 'horizon_xp')
        
        base_score = current_squad['horizon_xp'].sum()
        best: dict[str, Any] = {'action': 'HOLD', 'transfers_made': [], 'net_score': base_score, 'cost': 0}

        def check_team_limit(squad: pd.DataFrame, sell_ids: list[int], buy_teams: list[int]) -> bool:
            counts = squad['team_id'].value_counts().to_dict()
            for sid in sell_ids:
                team_rows = squad.loc[squad['id'] == sid, 'team_id']
                if team_rows.empty:
                    continue
                tid = team_rows.values[0]
                counts[tid] = counts.get(tid, 1) - 1
            for tid in buy_teams:
                if counts.get(tid, 0) >= 3:
                    return False
                counts[tid] = counts.get(tid, 0) + 1
            return True

        # Single transfers
        for _, sell in sell_cands.iterrows():
            buys = pool[(pool['pos_id'] == sell['pos_id']) & (pool['price'] <= bank + sell['price'])]
            for _, buy in buys.iterrows():
                if not check_team_limit(current_squad, [sell['id']], [buy['team_id']]):
                    continue
                gain = buy['horizon_xp'] - sell['horizon_xp']
                cost = 0 if free_transfers > 0 else 4
                if gain - cost > 0.5 and base_score + gain - cost > best['net_score']:
                    best = {'action': '1_TRANSFER', 'transfers_made': [(sell['name'], buy['name'])],
                            'net_score': base_score + gain - cost, 'cost': cost}

        # Double transfers
        print("Analyzing 2-Transfer options...")
        top_sell = current_squad.nsmallest(3, 'horizon_xp')
        top_pool = pool.head(20)
        
        for (i1, s1), (i2, s2) in itertools.combinations(top_sell.iterrows(), 2):
            combined_bank = bank + s1['price'] + s2['price']
            for _, b1 in top_pool[top_pool['pos_id'] == s1['pos_id']].iterrows():
                for _, b2 in top_pool[(top_pool['pos_id'] == s2['pos_id']) & 
                                      (top_pool['price'] <= combined_bank - b1['price']) &
                                      (top_pool['id'] != b1['id'])].iterrows():
                    if not check_team_limit(current_squad, [s1['id'], s2['id']], [b1['team_id'], b2['team_id']]):
                        continue
                    gain = (b1['horizon_xp'] + b2['horizon_xp']) - (s1['horizon_xp'] + s2['horizon_xp'])
                    cost = 0 if free_transfers >= 2 else 4 if free_transfers == 1 else 8
                    if gain - cost > best['net_score'] - base_score:
                        best = {'action': '2_TRANSFERS', 'transfers_made': [(s1['name'], b1['name']), (s2['name'], b2['name'])],
                                'net_score': base_score + gain - cost, 'cost': cost}
        return best

    def print_transfer_recommendation(self, rec: dict[str, Any]) -> None:
        header = pd.DataFrame([{'Action': rec['action'], 'Cost(pts)': f"-{rec['cost']}", 
                                'Projected(pts)': round(float(rec['net_score']), 2)}])
        print("\nTRANSFER SUMMARY")
        print(self._to_ascii_table(header))
        if rec['action'] != 'HOLD':
            print("\nTRANSFERS")
            print(self._to_ascii_table(pd.DataFrame([{'Sell': s, 'Buy': b} for s, b in rec['transfers_made']])))

    @staticmethod
    def _to_ascii_table(df: pd.DataFrame | None) -> str:
        """Render a bordered ASCII table."""
        if df is None or df.empty:
            return "(empty)"
        
        df = df.fillna('')
        cols = [str(c) for c in df.columns]
        rows = [[str(v) for v in row] for _, row in df.iterrows()]
        widths = [max(len(c), max((len(r[i]) for r in rows), default=0)) for i, c in enumerate(cols)]
        
        border = '+' + '+'.join('-' * (w + 2) for w in widths) + '+'
        def fmt(vals: list[str]) -> str:
            return '| ' + ' | '.join(str(v).ljust(w) for v, w in zip(vals, widths)) + ' |'
        
        return '\n'.join([border, fmt(cols), border] + [fmt(r) for r in rows] + [border])

    def format_squad(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format squad as a compact table."""
        out = df.copy()
        out['Team'] = out['team_id'].map(TEAM_MAP).fillna(out['team_id'])
        out['Pos'] = out['pos_id'].map(POS_MAP)
        out['XP(5)'] = out['horizon_xp'].round(1)
        out['XP(1)'] = out['next_gw_xp'].round(1)
        out['Price'] = out['price'].round(1)
        out['Role'] = out['is_captain_choice'].apply(lambda x: '(C)' if x == 1 else '') if 'is_captain_choice' in out.columns else ''
        return out[['name', 'Team', 'Pos', 'Price', 'XP(1)', 'XP(5)', 'Role']].rename(columns={'name': 'Name'})

    def display_squad(self, xi: pd.DataFrame, bench: pd.DataFrame, cap_id: int, vice_id: int) -> None:
        print("\nSTARTING XI")
        disp = self.format_squad(xi).copy()
        disp.loc[xi['id'] == cap_id, 'Role'] = 'C'
        disp.loc[xi['id'] == vice_id, 'Role'] = 'VC'
        print(self._to_ascii_table(disp))
        print("\nSUBSTITUTES")
        print(self._to_ascii_table(self.format_squad(bench)))