import pandas as pd
import numpy as np
import pulp
import itertools
from scipy.optimize import minimize

POINTS_MAP = {
    1: {'goal': 6, 'assist': 3, 'cs': 4},
    2: {'goal': 6, 'assist': 3, 'cs': 4},
    3: {'goal': 5, 'assist': 3, 'cs': 1},
    4: {'goal': 4, 'assist': 3, 'cs': 0}
}
SQUAD_COUNTS = {1: 2, 2: 5, 3: 5, 4: 3}
MIN_STARTERS = {1: 1, 2: 3, 3: 2, 4: 1}
POS_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
TEAM_MAP = {
    1: "Arsenal", 2: "Aston Villa", 3: "Burnley", 4: "Bournemouth",
    5: "Brentford", 6: "Brighton", 7: "Chelsea", 8: "Crystal Palace",
    9: "Everton", 10: "Fulham", 11: "Leeds", 12: "Liverpool",
    13: "Man City", 14: "Man Utd", 15: "Newcastle", 16: "Nott'm Forest",
    17: "Sunderland", 18: "Spurs", 19: "West Ham", 20: "Wolves"
}

class FPLEngine:
    def __init__(self, fixtures_df, history_df, players_df):
        self.fixtures = fixtures_df
        self.history = history_df
        self.players = players_df
        self.id_map = self.players.set_index('id')['team'].to_dict()
        if 'team' not in self.history.columns:
            self.history['team'] = self.history['player_id'].map(self.id_map)

    def _solve_poisson_regression(self, df):
        """Calculate team attack/defense strengths via Poisson regression."""
        teams = sorted(set(df['team_h'].unique()) | set(df['team_a'].unique()))
        n_teams = len(teams)

        def loss(params):
            home_adv, intercept = params[0], params[1]
            alphas = dict(zip(teams, params[2:2+n_teams]))
            betas = dict(zip(teams, params[2+n_teams:]))
            mu_h = np.exp(intercept + home_adv + df['team_h'].map(alphas) + df['team_a'].map(betas))
            mu_a = np.exp(intercept + df['team_a'].map(alphas) + df['team_h'].map(betas))
            ll = (df['team_h_score'] * np.log(mu_h) - mu_h) + (df['team_a_score'] * np.log(mu_a) - mu_a)
            return -np.sum(ll)

        init_params = np.concatenate([[0.2, 0.0], np.zeros(n_teams), np.zeros(n_teams)])
        res = minimize(loss, init_params, method='L-BFGS-B', options={'disp': False, 'maxiter': 100})
        params = res.x
        return params[0], params[1], dict(zip(teams, params[2:2+n_teams])), dict(zip(teams, params[2+n_teams:]))

    def _calculate_nailedness(self):
        """Calculate minute security factor per player."""
        season_stats = self.history.groupby('player_id')['minutes'].agg(['sum'])
        team_max_mins = {}
        for pid, row in season_stats.iterrows():
            tid = self.id_map.get(pid)
            team_max_mins[tid] = max(team_max_mins.get(tid, 0), row['sum'])

        nailed_map = {}
        for pid, row in season_stats.iterrows():
            share = row['sum'] / max(1, team_max_mins.get(self.id_map.get(pid), 90))
            nailed_map[pid] = 1.05 if share >= 0.90 else 1.00 if share >= 0.75 else 0.90 if share >= 0.50 else 0.75
        return nailed_map

    def _calculate_usage_stats(self, current_gw):
        """Calculate player ability using xGI with Bayesian shrinkage."""
        season_window = self.history[self.history['round'] >= current_gw - 38].copy()
        
        # Identify top xGI player per team/position
        talisman_bonus = {}
        pid_to_pos = self.players.set_index('id')['element_type'].to_dict()
        season_window['pos'] = season_window['player_id'].map(pid_to_pos)
        
        for (tid, pos), group in season_window.groupby(['team', 'pos']):
            agg = group.groupby('player_id')['expected_goal_involvements'].sum()
            if not agg.empty:
                talisman_bonus[agg.idxmax()] = 1.05

        def bayesian_avg(total_val, total_mins, avg_rate, dummy_mins=400):
            return (total_val + (avg_rate * (dummy_mins/90))) / ((total_mins + dummy_mins) / 90)

        AVG_XG, AVG_XA = 0.12, 0.10
        player_stats = {}
        
        for pid in season_window['player_id'].unique():
            p_data = season_window[season_window['player_id'] == pid]
            mins = p_data['minutes'].sum()
            eff_mins = max(mins, 90)
            
            total_xg = p_data.get('expected_goals', pd.Series([0]*len(p_data))).sum()
            total_xa = p_data.get('expected_assists', pd.Series([0]*len(p_data))).sum()
            total_cbit = p_data['clearances_blocks_interceptions'].sum() + p_data['tackles'].sum()
            
            bonus = talisman_bonus.get(pid, 1.0)
            player_stats[pid] = {
                'threat_share': bayesian_avg(total_xg, mins, AVG_XG) * 100 * bonus,
                'create_share': bayesian_avg(total_xa, mins, AVG_XA) * 100 * bonus,
                'cbit_share': (total_cbit / eff_mins) * 90,
                'recovery_share': (p_data['recoveries'].sum() / eff_mins) * 90,
                'saves_p90': (p_data['saves'].sum() / eff_mins) * 90,
                'team': self.id_map.get(pid)
            }
        return player_stats

    def _get_team_context(self, team_id, player_stats):
        """Get team's total threat/creativity from top 14 players."""
        team_players = [p for p in player_stats.values() if p['team'] == team_id]
        if not team_players:
            return 100.0, 100.0
        top_squad = sorted(team_players, key=lambda x: x['threat_share'] + x['create_share'], reverse=True)[:14]
        return max(10, sum(p['threat_share'] for p in top_squad)), max(10, sum(p['create_share'] for p in top_squad))

    def _get_team_defensive_context(self, team_id, player_stats):
        """Get team's total defensive contribution."""
        team_players = [p for p in player_stats.values() if p['team'] == team_id]
        if not team_players:
            return 1.0
        return max(1, sum(p['cbit_share'] for p in sorted(team_players, key=lambda x: x['cbit_share'], reverse=True)[:11]))

    def train_and_predict(self, current_gw, horizon=3):
        # 1. POISSON MODEL (Team Strength)
        train_df = self.fixtures[(self.fixtures['event'] < current_gw) & (self.fixtures['finished'] == True)].copy()
        if len(train_df) < 20:
            home_adv, intercept, alphas, betas = 0.2, 0.0, {}, {}
        else:
            home_adv, intercept, alphas, betas = self._solve_poisson_regression(train_df)

        # 2. PREDICT FIXTURES
        horizon_gws = list(range(current_gw, current_gw + horizon))
        future_fix = self.fixtures[self.fixtures['event'].isin(horizon_gws)].copy()
        
        fix_preds = []
        for _, row in future_fix.iterrows():
            h, a = row['team_h'], row['team_a']
            att_h, def_h = alphas.get(h, 0), betas.get(h, 0)
            att_a, def_a = alphas.get(a, 0), betas.get(a, 0)
            
            lambda_h = np.exp(intercept + home_adv + att_h + def_a)
            lambda_a = np.exp(intercept + att_a + def_h)
            
            fix_preds.append({
                'event': row['event'], 'team': h, 'opp': a, 
                'goals_scored': lambda_h, 'goals_conceded': lambda_a,
                'clean_sheet_prob': np.exp(-lambda_a), 'opp_pressure': lambda_a
            })
            fix_preds.append({
                'event': row['event'], 'team': a, 'opp': h, 
                'goals_scored': lambda_a, 'goals_conceded': lambda_h,
                'clean_sheet_prob': np.exp(-lambda_h), 'opp_pressure': lambda_h
            })
            
        df_fix = pd.DataFrame(fix_preds)
        if df_fix.empty: return pd.DataFrame()

        # 3. PLAYER PREDICTIONS
        usage_stats = self._calculate_usage_stats(current_gw)
        nailed_map = self._calculate_nailedness() 
        past_hist = self.history[self.history['round'] < current_gw]
        projections = []
        
        for _, p in self.players.iterrows():
            pid = p['id']
            # Injury/Status Check
            chance = p.get('chance_of_playing_next_round', 100)
            if pd.isna(chance): chance = 100
            if p['status'] in ['s', 'n'] or chance < 25: continue

            # Min Minutes Filter
            p_hist = past_hist[past_hist['player_id'] == pid]
            # Allow new players if we have no history, but penalize slightly
            if len(p_hist) == 0: 
                role_mins = 60 # Optimistic assumption for new signings
            else:
                # Use recent average minutes
                role_mins = p_hist.tail(5)['minutes'].mean()
            
            if role_mins < 30 and chance < 100: continue

            prob_play = chance / 100.0
            effective_prob = min(1.0, prob_play * nailed_map.get(pid, 0.75))
            
            # Get Stats
            p_stats = usage_stats.get(pid, {
                'threat_share': 5, 'create_share': 5, 'cbit_share': 1, 'recovery_share': 1, 'saves_p90': 0
            })
            
            # Calculate Market Share of Team's output
            team_threat, team_create = self._get_team_context(p['team'], usage_stats)
            goal_share = min(p_stats['threat_share'] / team_threat, 0.60) # Cap at 60% of team goals
            assist_share = min(p_stats['create_share'] / team_create, 0.50)
            
            team_cbit_total = self._get_team_defensive_context(p['team'], usage_stats)
            cbit_share_ratio = p_stats['cbit_share'] / team_cbit_total

            team_fix = df_fix[df_fix['team'] == p['team']]
            total_xp, next_gw_xp = 0, 0
            
            for _, f in team_fix.iterrows():
                play_factor = role_mins / 90.0
                
                # Attacking Points (Using xGI share)
                xp_goals = f['goals_scored'] * goal_share * play_factor * POINTS_MAP[p['element_type']]['goal']
                xp_assists = f['goals_scored'] * assist_share * play_factor * POINTS_MAP[p['element_type']]['assist']
                
                # Defensive Points
                xp_clean_sheet = 0
                xp_conceded = 0
                if role_mins >= 60:
                    xp_clean_sheet = POINTS_MAP[p['element_type']]['cs'] * f['clean_sheet_prob']
                if p['element_type'] in [1, 2]:
                    xp_conceded = (f['goals_conceded'] / 2) * -1
                
                # Bonus/BPS proxies (CBIT + Saves)
                base_team_cbit = 40.0 
                pressure_factor = f['opp_pressure'] / 1.3
                est_player_cbit = base_team_cbit * pressure_factor * cbit_share_ratio * play_factor
                
                xp_bps = 0
                if p['element_type'] == 2: 
                    # Defenders get BPS for defensive actions
                    xp_bps = 2 * (1 / (1 + np.exp(-(est_player_cbit - 10))))
                elif p['element_type'] in [3, 4]: 
                    est_recov = p_stats['recovery_share'] * play_factor
                    xp_bps = 2 * (1 / (1 + np.exp(-(est_player_cbit + est_recov - 12))))

                xp_saves = 0
                if p['element_type'] == 1:
                    xp_saves = ((f['goals_conceded'] * 3.5 * 0.70 * play_factor) / 3) * 1

                xp_app = (2 if role_mins >= 60 else 1)
                gw_xp = (xp_goals + xp_assists + xp_clean_sheet + xp_conceded + xp_bps + xp_saves + xp_app) * effective_prob
                
                total_xp += gw_xp
                if f['event'] == current_gw: next_gw_xp = gw_xp
            
            projections.append({
                'id': pid, 'name': p['web_name'], 'team_id': p['team'], 'pos_id': p['element_type'],
                'price': p['now_cost']/10.0, 'horizon_xp': total_xp, 'next_gw_xp': next_gw_xp
            })
            
        return pd.DataFrame(projections)

    def optimize_squad(self, df, budget=100.0):
        """Standard Linear Optimization for Squad Selection"""
        
        prob = pulp.LpProblem("FPL_Squad", pulp.LpMaximize)
        ids = df['id'].tolist()
        
        x = pulp.LpVariable.dicts("player", ids, cat='Binary')
        c = pulp.LpVariable.dicts("captain", ids, cat='Binary')
        
        prob += pulp.lpSum([df[df['id']==i]['horizon_xp'].values[0] * (x[i] + c[i]) for i in ids])
        prob += pulp.lpSum([df[df['id']==i]['price'].values[0] * x[i] for i in ids]) <= budget
        prob += pulp.lpSum([x[i] for i in ids]) == 15
        
        for pos, count in SQUAD_COUNTS.items():
            prob += pulp.lpSum([x[i] for i in ids if df[df['id']==i]['pos_id'].values[0] == pos]) == count
            
        for t in df['team_id'].unique():
            prob += pulp.lpSum([x[i] for i in ids if df[df['id']==i]['team_id'].values[0] == t]) <= 3
            
        prob += pulp.lpSum([c[i] for i in ids]) == 1
        for i in ids:
            prob += c[i] <= x[i]

        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[prob.status] != 'Optimal': return pd.DataFrame()
        
        selected_ids = [i for i in ids if x[i].varValue == 1]
        captain_id = [i for i in ids if c[i].varValue == 1][0]
        
        result_df = df[df['id'].isin(selected_ids)].copy()
        result_df['is_captain_choice'] = result_df['id'].apply(lambda pid: 1 if pid == captain_id else 0)
        return result_df

    def pick_team_sheet(self, squad_df):
        """Picks Best Starting XI from the 15 selected players"""
        squad_df = squad_df.sort_values('next_gw_xp', ascending=False)
        
        gks = squad_df[squad_df['pos_id'] == 1]
        defs = squad_df[squad_df['pos_id'] == 2]
        mids = squad_df[squad_df['pos_id'] == 3]
        fwds = squad_df[squad_df['pos_id'] == 4]
        
        starter_ids = []
        starter_ids.extend(gks.head(MIN_STARTERS[1])['id'].tolist())
        starter_ids.extend(defs.head(MIN_STARTERS[2])['id'].tolist())
        starter_ids.extend(mids.head(MIN_STARTERS[3])['id'].tolist())
        starter_ids.extend(fwds.head(MIN_STARTERS[4])['id'].tolist())
        
        remaining = squad_df[~squad_df['id'].isin(starter_ids)]
        remaining_outfield = remaining[remaining['pos_id'] != 1]
        
        flex_starters = remaining_outfield.head(4)
        starter_ids.extend(flex_starters['id'].tolist())
        
        starting_xi = squad_df[squad_df['id'].isin(starter_ids)].sort_values('pos_id')
        bench = squad_df[~squad_df['id'].isin(starter_ids)].sort_values('pos_id')
        
        solver_cap = squad_df[squad_df.get('is_captain_choice', 0) == 1]
        if not solver_cap.empty and solver_cap.iloc[0]['id'] in starting_xi['id'].values:
            captain_id = solver_cap.iloc[0]['id']
            remain = starting_xi[starting_xi['id'] != captain_id].sort_values('next_gw_xp', ascending=False)
            vice_id = remain.iloc[0]['id']
        else:
            cap_candidates = starting_xi.sort_values('next_gw_xp', ascending=False)
            captain_id = cap_candidates.iloc[0]['id']
            vice_id = cap_candidates.iloc[1]['id']
            
        return starting_xi, bench, captain_id, vice_id

    def recommend_transfers(self, current_squad, bank, free_transfers=1):
        """Find best transfer options using beam search."""
        print("\nAnalyzing transfer market...")
        
        all_projections = self.train_and_predict(38)
        current_ids = current_squad['id'].tolist()
        available_pool = all_projections[~all_projections['id'].isin(current_ids)].sort_values('horizon_xp', ascending=False).head(50)
        sell_candidates = current_squad.sort_values('horizon_xp', ascending=True).head(5)
        
        best_move = {'action': 'HOLD', 'transfers_made': [], 'net_score': current_squad['horizon_xp'].sum(), 'cost': 0}
        
        print(f"Analyzing HOLD (Score: {best_move['net_score']:.1f})")
        print("Analyzing 1-Transfer options...")
        
        for _, sell_p in sell_candidates.iterrows():
            current_bank = bank + sell_p['price']
            valid_buys = available_pool[
                (available_pool['pos_id'] == sell_p['pos_id']) & 
                (available_pool['price'] <= current_bank)
            ]
            
            for _, buy_p in valid_buys.iterrows():
                current_team_counts = current_squad['team_id'].value_counts()
                buy_team_count = current_team_counts.get(buy_p['team_id'], 0)
                if sell_p['team_id'] == buy_p['team_id']: buy_team_count -= 1
                
                if buy_team_count >= 3: continue 
                
                gain = buy_p['horizon_xp'] - sell_p['horizon_xp']
                cost = 0 if free_transfers > 0 else 4
                net_improvement = gain - cost
                
                if net_improvement > 0.5:
                    total_score = current_squad['horizon_xp'].sum() + net_improvement
                    if total_score > best_move['net_score']:
                        best_move = {
                            'action': '1_TRANSFER',
                            'transfers_made': [(sell_p['name'], buy_p['name'])],
                            'net_score': total_score,
                            'cost': cost
                        }

        print("Analyzing 2-Transfer options...")
        sell_pair_candidates = current_squad.sort_values('horizon_xp', ascending=True).head(3)
        buy_pool_limited = available_pool.head(20)
        sell_indices = list(itertools.combinations(sell_pair_candidates.index, 2))
        
        for idx1, idx2 in sell_indices:
            s1 = current_squad.loc[idx1]
            s2 = current_squad.loc[idx2]
            combined_bank = bank + s1['price'] + s2['price']
            
            replacements_s1 = buy_pool_limited[(buy_pool_limited['pos_id'] == s1['pos_id'])]
            
            for _, b1 in replacements_s1.iterrows():
                remaining_bank = combined_bank - b1['price']
                replacements_s2 = buy_pool_limited[
                    (buy_pool_limited['pos_id'] == s2['pos_id']) &
                    (buy_pool_limited['price'] <= remaining_bank) &
                    (buy_pool_limited['id'] != b1['id'])
                ]
                
                for _, b2 in replacements_s2.iterrows():
                    team_counts = current_squad['team_id'].value_counts().to_dict()
                    team_counts[s1['team_id']] = team_counts.get(s1['team_id'], 0) - 1
                    team_counts[s2['team_id']] = team_counts.get(s2['team_id'], 0) - 1
                    
                    if team_counts.get(b1['team_id'], 0) >= 3: continue
                    team_counts[b1['team_id']] = team_counts.get(b1['team_id'], 0) + 1
                    if team_counts.get(b2['team_id'], 0) >= 3: continue
                    
                    gain = (b1['horizon_xp'] + b2['horizon_xp']) - (s1['horizon_xp'] + s2['horizon_xp'])
                    hit_cost = 0
                    if free_transfers == 1: hit_cost = 4
                    elif free_transfers == 0: hit_cost = 8
                    
                    net_improvement = gain - hit_cost
                    if net_improvement > best_move['net_score'] - current_squad['horizon_xp'].sum():
                         best_move = {
                            'action': '2_TRANSFERS',
                            'transfers_made': [(s1['name'], b1['name']), (s2['name'], b2['name'])],
                            'net_score': current_squad['horizon_xp'].sum() + net_improvement,
                            'cost': hit_cost
                        }

        return best_move

    def print_transfer_recommendation(self, rec):
        print("\n" + "="*40)
        print(f"RECOMMENDATION: {rec['action']}")
        print("="*40)
        print(f"Cost (Hit): -{rec['cost']} pts")
        print(f"Projected Score: {rec['net_score']:.2f} pts")
        
        if rec['action'] == 'HOLD':
            print("\nAdvice: Roll your transfer.")
        else:
            print("\nSuggested Moves:")
            for sell, buy in rec['transfers_made']:
                print(f"  SELL: {sell} -> BUY: {buy}")

    def format_squad(self, df):
        out = df.copy()
        out['team'] = out['team_id'].map(TEAM_MAP).fillna(out['team_id'])
        out['position'] = out['pos_id'].map(POS_MAP)
        out['horizon_xp'] = out['horizon_xp'].astype(float).round(1)
        out['next_gw_xp'] = out['next_gw_xp'].astype(float).round(1)
        out['role'] = ''
        if 'is_captain_choice' in out.columns:
            out.loc[out['is_captain_choice'] == 1, 'role'] = '(C)'
        return out[['name', 'team', 'position', 'price', 'next_gw_xp', 'horizon_xp', 'role']]

    def display_squad(self, starting_xi, bench, cap_id, vice_id):
        print("\n--- STARTING XI ---")
        disp_start = self.format_squad(starting_xi).copy()
        disp_start.loc[starting_xi['id'] == cap_id, 'role'] = 'CAPTAIN'
        disp_start.loc[starting_xi['id'] == vice_id, 'role'] = 'VICE-CAP'
        print(disp_start.to_string(index=False))
        
        print("\n--- SUBSTITUTES ---")
        disp_bench = self.format_squad(bench)
        print(disp_bench.to_string(index=False))