import pandas as pd
import numpy as np
import pulp
from scipy.optimize import minimize

# FPL scoring rules: {position: {action: points}}
POINTS_MAP = {
    1: {'goal': 6, 'assist': 3, 'cs': 4},  # GK
    2: {'goal': 6, 'assist': 3, 'cs': 4},  # DEF
    3: {'goal': 5, 'assist': 3, 'cs': 1},  # MID
    4: {'goal': 4, 'assist': 3, 'cs': 0}   # FWD
}

SQUAD_COUNTS = {1: 2, 2: 5, 3: 5, 4: 3}      # Players per position in squad
STARTING_COUNTS = {1: 1, 2: 4, 3: 4, 4: 2}   # Players per position in starting XI
POS_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
TEAM_MAP = {
    1: "Arsenal", 2: "Aston Villa", 3: "Burnley", 4: "Bournemouth",
    5: "Brentford", 6: "Brighton", 7: "Chelsea", 8: "Crystal Palace",
    9: "Everton", 10: "Fulham", 11: "Leeds", 12: "Liverpool",
    13: "Man City", 14: "Man Utd", 15: "Newcastle", 16: "Nott'm Forest",
    17: "Sunderland", 18: "Spurs", 19: "West Ham", 20: "Wolves"
}

class FPLEngine:
    def __init__(self, fixtures_path, history_path, players_path):
        self.fixtures = pd.read_csv(fixtures_path)
        self.history = pd.read_csv(history_path)
        self.players = pd.read_csv(players_path)
        self.id_map = self.players.set_index('id')['team'].to_dict()
        self.history['team'] = self.history['player_id'].map(self.id_map)

    def _solve_poisson_regression(self, df):
        """Calculate team attack/defense strengths using Poisson regression."""
        teams = sorted(set(df['team_h'].unique()) | set(df['team_a'].unique()))
        n = len(teams)

        def loss(params):
            home_adv, intercept = params[0], params[1]
            alphas, betas = dict(zip(teams, params[2:2+n])), dict(zip(teams, params[2+n:]))
            mu_h = np.exp(intercept + home_adv + df['team_h'].map(alphas) + df['team_a'].map(betas))
            mu_a = np.exp(intercept + df['team_a'].map(alphas) + df['team_h'].map(betas))
            return -np.sum((df['team_h_score'] * np.log(mu_h) - mu_h) + 
                          (df['team_a_score'] * np.log(mu_a) - mu_a))

        init = np.concatenate([[0.2, 0.0], np.zeros(2 * n)])
        res = minimize(loss, init, method='L-BFGS-B', options={'disp': False, 'maxiter': 100})
        p = res.x
        return p[0], p[1], dict(zip(teams, p[2:2+n])), dict(zip(teams, p[2+n:]))

    def _calculate_nailedness(self):
        """Classify players into tiers based on minutes share."""
        stats = self.history.groupby('player_id')['minutes'].sum()
        
        # Find max minutes per team
        team_max = {}
        for pid, mins in stats.items():
            tid = self.id_map.get(pid)
            team_max[tid] = max(team_max.get(tid, 0), mins)

        # Assign tier factors based on share
        nailed = {}
        for pid, mins in stats.items():
            share = mins / max(1, team_max.get(self.id_map.get(pid), 90))
            nailed[pid] = 1.05 if share >= 0.90 else 1.0 if share >= 0.70 else 0.90 if share >= 0.50 else 0.75
        return nailed

    def _calculate_usage_stats(self, current_gw):
        """Calculate usage stats using depth chart logic to penalize bench players."""
        season_df = self.history[self.history['round'] >= current_gw - 38].copy()
        form_df = self.history[self.history['round'] >= current_gw - 5].copy()
        pid_to_pos = self.players.set_index('id')['element_type'].to_dict()
        season_df['pos'] = season_df['player_id'].map(pid_to_pos)

        # Build depth charts: (team, pos) -> sorted PP90 scores
        depth_charts = {}
        for (tid, pos), grp in season_df.groupby(['team', 'pos']):
            scores = []
            for pid, pdata in grp.groupby('player_id'):
                mins = pdata['minutes'].sum()
                if mins > 300:
                    scores.append((pdata['total_points'].sum() / mins) * 90)
            depth_charts[(tid, pos)] = sorted(scores, reverse=True)

        def get_stats(df, pid, team_id, pos_id):
            """Extract per-90 stats with depth chart penalty for transfers."""
            mins = df['minutes'].sum()
            if mins < 90:
                return None

            my_pp90 = (df['total_points'].sum() / mins) * 90
            incumbents = depth_charts.get((team_id, pos_id), [])
            better_count = sum(1 for s in incumbents if s > my_pp90 + 0.1)
            
            # Penalty based on depth chart rank
            penalty = {0: 1.0, 1: 0.90, 2: 0.75}.get(better_count, 0.50)

            threats, creates, cbits, saves = [], [], [], []
            for _, row in df.iterrows():
                mult = penalty if row.get('team', team_id) != team_id else 1.0
                threats.append(row['threat'] * mult)
                creates.append(row['creativity'] * mult)
                cbits.append((row['clearances_blocks_interceptions'] + row['tackles']) * mult)
                saves.append(row['saves'] * mult)

            return {
                'threat': np.sum(threats) / mins * 90,
                'create': np.sum(creates) / mins * 90,
                'cbit': np.sum(cbits) / mins * 90,
                'recovery': df['recoveries'].sum() / mins * 90,
                'saves': np.sum(saves) / mins * 90,
                'variance': df['total_points'].std()
            }

        # Process each player
        player_stats = {}
        for pid in season_df['player_id'].unique():
            team, pos = self.id_map.get(pid), pid_to_pos.get(pid, 3)
            s = get_stats(season_df[season_df['player_id'] == pid], pid, team, pos)
            f = get_stats(form_df[form_df['player_id'] == pid], pid, team, pos)
            if not s:
                continue

            # Blend season (60%) and form (40%)
            if s is not None and f is not None:
                blend = lambda sk, fk, s_=s, f_=f: s_[sk] * 0.6 + f_[fk] * 0.4
                threat, create = blend('threat', 'threat'), blend('create', 'create')
                cbit, recovery, saves = blend('cbit', 'cbit'), blend('recovery', 'recovery'), blend('saves', 'saves')
            elif s is not None:
                threat, create, cbit, recovery, saves = s['threat'], s['create'], s['cbit'], s['recovery'], s['saves']
            else:
                continue

            # Consistency adjustment
            vol = s['variance'] if not pd.isna(s['variance']) else 3.0
            cons = 0.90 if vol > 4.5 else 1.05 if vol < 2.5 else 1.0

            player_stats[pid] = {
                'threat_share': threat * cons, 'create_share': create * cons,
                'cbit_share': cbit, 'recovery_share': recovery, 'saves_p90': saves, 'team': team
            }
        return player_stats

    def _get_team_context(self, team_id, stats):
        """Get total attacking output of team's top 11."""
        players = [p for p in stats.values() if p['team'] == team_id]
        if not players:
            return 100.0, 100.0
        top11 = sorted(players, key=lambda x: x['threat_share'] + x['create_share'], reverse=True)[:11]
        return max(1, sum(p['threat_share'] for p in top11)), max(1, sum(p['create_share'] for p in top11))

    def _get_team_defensive_context(self, team_id, stats):
        """Get total defensive workload of team's top 11."""
        players = [p for p in stats.values() if p['team'] == team_id]
        if not players:
            return 1.0
        top11 = sorted(players, key=lambda x: x['cbit_share'], reverse=True)[:11]
        return max(1, sum(p['cbit_share'] for p in top11))

    def train_and_predict(self, current_gw, horizon=3):
        """Train Poisson model and generate player predictions."""
        # Train team model
        train_df = self.fixtures[(self.fixtures['event'] < current_gw) & self.fixtures['finished']].copy()
        if len(train_df) < 20:
            home_adv, intercept, alphas, betas = 0.2, 0.0, {}, {}
        else:
            home_adv, intercept, alphas, betas = self._solve_poisson_regression(train_df)

        # Generate match predictions
        gws = list(range(current_gw, current_gw + horizon))
        future = self.fixtures[self.fixtures['event'].isin(gws)]

        fix_preds = []
        for _, row in future.iterrows():
            h, a = row['team_h'], row['team_a']
            lam_h = np.exp(intercept + home_adv + alphas.get(h, 0) + betas.get(a, 0))
            lam_a = np.exp(intercept + alphas.get(a, 0) + betas.get(h, 0))
            
            fix_preds.append({'event': row['event'], 'team': h, 'opp': a,
                'goals_scored': lam_h, 'goals_conceded': lam_a,
                'clean_sheet_prob': np.exp(-lam_a), 'opp_pressure': lam_a})
            fix_preds.append({'event': row['event'], 'team': a, 'opp': h,
                'goals_scored': lam_a, 'goals_conceded': lam_h,
                'clean_sheet_prob': np.exp(-lam_h), 'opp_pressure': lam_h})

        df_fix = pd.DataFrame(fix_preds)
        if df_fix.empty:
            return pd.DataFrame()

        # Player predictions
        usage = self._calculate_usage_stats(current_gw)
        nailed = self._calculate_nailedness()
        past = self.history[self.history['round'] < current_gw]
        projections = []

        for _, p in self.players.iterrows():
            pid, chance = p['id'], p.get('chance_of_playing_next_round', 100)
            if pd.isna(chance):
                chance = 100
            if p['status'] in ['s', 'n'] or chance < 25:
                continue

            hist = past[past['player_id'] == pid]
            active = hist[hist['minutes'] > 0]
            if len(active) < 3:
                continue

            role_mins = active.tail(5)['minutes'].mean()
            if role_mins < 30:
                continue

            # Effective play probability
            eff_prob = min(1.0, (chance / 100.0) * nailed.get(pid, 0.75))
            if chance == 75:
                role_mins *= 0.8

            # Player and team context
            ps = usage.get(pid, {'threat_share': 5, 'create_share': 5, 'cbit_share': 1, 'recovery_share': 1, 'saves_p90': 0})
            t_threat, t_create = self._get_team_context(p['team'], usage)
            goal_share = min(ps['threat_share'] / t_threat, 0.50)
            assist_share = min(ps['create_share'] / t_create, 0.50)
            cbit_ratio = ps['cbit_share'] / self._get_team_defensive_context(p['team'], usage)

            # Simulate matches
            total_xp, next_gw_xp = 0, 0
            pos = p['element_type']
            pts = POINTS_MAP[pos]

            for _, f in df_fix[df_fix['team'] == p['team']].iterrows():
                pf = role_mins / 90.0  # play factor

                # Attack points
                xp = f['goals_scored'] * pf * (goal_share * pts['goal'] + assist_share * pts['assist'])

                # Defense points
                if role_mins >= 60:
                    xp += pts['cs'] * f['clean_sheet_prob']
                if pos in [1, 2]:
                    xp -= f['goals_conceded'] / 2

                # Defensive work bonus (CBIT)
                est_cbit = 40.0 * (f['opp_pressure'] / 1.3) * cbit_ratio * pf
                if pos == 2:
                    xp += 2 / (1 + np.exp(-(est_cbit - 10)))
                elif pos in [3, 4]:
                    xp += 2 / (1 + np.exp(-(est_cbit + ps['recovery_share'] * pf - 12)))

                # Saves (GK only)
                if pos == 1:
                    xp += (f['goals_conceded'] * 3.5 * 0.70 * pf) / 3

                # Appearance points
                xp += 2 if role_mins >= 60 else 1

                gw_xp = xp * eff_prob
                total_xp += gw_xp
                if f['event'] == current_gw:
                    next_gw_xp = gw_xp

            projections.append({
                'id': pid, 'name': p['web_name'], 'team_id': p['team'], 'pos_id': pos,
                'price': p['now_cost'] / 10.0, 'horizon_xp': total_xp, 'next_gw_xp': next_gw_xp
            })

        return pd.DataFrame(projections)

    def optimize_squad(self, df, budget=100.0):
        """Select optimal 15-player squad using linear programming."""
        prob = pulp.LpProblem("FPL_Squad", pulp.LpMaximize)
        ids = df['id'].tolist()
        data = df.set_index('id')

        x = pulp.LpVariable.dicts("player", ids, cat='Binary')
        c = pulp.LpVariable.dicts("captain", ids, cat='Binary')

        # Maximize points (captain gets double)
        prob += pulp.lpSum([data.loc[i, 'horizon_xp'] * (x[i] + c[i]) for i in ids])

        # Constraints
        prob += pulp.lpSum([data.loc[i, 'price'] * x[i] for i in ids]) <= budget
        prob += pulp.lpSum([x[i] for i in ids]) == 15
        prob += pulp.lpSum([c[i] for i in ids]) == 1

        for pos, cnt in SQUAD_COUNTS.items():
            prob += pulp.lpSum([x[i] for i in ids if data.loc[i, 'pos_id'] == pos]) == cnt
        for t in df['team_id'].unique():
            prob += pulp.lpSum([x[i] for i in ids if data.loc[i, 'team_id'] == t]) <= 3
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

    def pick_team_sheet(self, squad_df):
        """Select starting XI and bench from squad."""
        squad = squad_df.sort_values('next_gw_xp', ascending=False)

        # Pick best players per position
        starters = pd.concat([squad[squad['pos_id'] == p].head(c) for p, c in STARTING_COUNTS.items()])
        bench = squad[~squad['id'].isin(starters['id'])].sort_values('next_gw_xp', ascending=False)
        starters = starters.sort_values('pos_id')

        # Captain selection
        cap_choice = squad[squad.get('is_captain_choice', 0) == 1]
        if not cap_choice.empty and cap_choice.iloc[0]['id'] in starters['id'].values:
            cap_id = cap_choice.iloc[0]['id']
            vc_id = starters[starters['id'] != cap_id].sort_values('next_gw_xp', ascending=False).iloc[0]['id']
        else:
            top = starters.sort_values('next_gw_xp', ascending=False)
            cap_id, vc_id = top.iloc[0]['id'], top.iloc[1]['id']

        return starters, bench, cap_id, vc_id

    def format_squad(self, df):
        """Format squad data for display."""
        out = df.copy()
        out['team'] = out['team_id'].map(TEAM_MAP).fillna(out['team_id'])
        out['position'] = out['pos_id'].map(POS_MAP)
        out['horizon_xp'] = out['horizon_xp'].round(1)
        out['next_gw_xp'] = out['next_gw_xp'].round(1)
        out['role'] = out.get('is_captain_choice', 0).apply(lambda x: '(C)' if x == 1 else '')
        return out[['name', 'team', 'position', 'price', 'next_gw_xp', 'horizon_xp', 'role']]