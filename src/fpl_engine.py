import pandas as pd
import numpy as np
import pulp
from scipy.optimize import minimize

# Scoring rules by position
POINTS_MAP = {
    1: {'goal': 6, 'assist': 3, 'cs': 4},  # GK
    2: {'goal': 6, 'assist': 3, 'cs': 4},  # DEF
    3: {'goal': 5, 'assist': 3, 'cs': 1},  # MID
    4: {'goal': 4, 'assist': 3, 'cs': 0}   # FWD
}

SQUAD_COUNTS = {1: 2, 2: 5, 3: 5, 4: 3}
STARTING_COUNTS = {1: 1, 2: 4, 3: 4, 4: 2}

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
        
        id_map = self.players.set_index('id')['team'].to_dict()
        self.history['team'] = self.history['player_id'].map(id_map)

    def _solve_poisson_regression(self, df):
        """Fit team attack/defense strengths using Poisson regression."""
        teams = sorted(set(df['team_h']) | set(df['team_a']))
        n = len(teams)

        def loss(params):
            home_adv, intercept = params[0], params[1]
            alphas = dict(zip(teams, params[2:2+n]))
            betas = dict(zip(teams, params[2+n:]))
            
            mu_h = np.exp(intercept + home_adv + df['team_h'].map(alphas) + df['team_a'].map(betas))
            mu_a = np.exp(intercept + df['team_a'].map(alphas) + df['team_h'].map(betas))
            
            ll = (df['team_h_score'] * np.log(mu_h) - mu_h) + (df['team_a_score'] * np.log(mu_a) - mu_a)
            return -np.sum(ll)

        init = np.concatenate([[0.2, 0.0], np.zeros(2 * n)])
        res = minimize(loss, init, method='L-BFGS-B', options={'maxiter': 100})
        
        p = res.x
        return p[0], p[1], dict(zip(teams, p[2:2+n])), dict(zip(teams, p[2+n:]))

    def train_and_predict(self, current_gw, horizon=3):
        """Train model and predict player points."""
        # Train on completed fixtures
        train = self.fixtures[(self.fixtures['event'] < current_gw) & self.fixtures['finished']]
        
        if len(train) < 20:
            home_adv, intercept, alphas, betas = 0.2, 0.0, {}, {}
        else:
            home_adv, intercept, alphas, betas = self._solve_poisson_regression(train)

        # Get upcoming fixtures
        gws = range(current_gw, current_gw + horizon)
        future = self.fixtures[self.fixtures['event'].isin(gws)]
        
        if future.empty:
            return pd.DataFrame()

        # Predict match outcomes
        fix_preds = []
        for _, r in future.iterrows():
            h, a = r['team_h'], r['team_a']
            lambda_h = np.exp(intercept + home_adv + alphas.get(h, 0) + betas.get(a, 0))
            lambda_a = np.exp(intercept + alphas.get(a, 0) + betas.get(h, 0))
            
            fix_preds.append({'event': r['event'], 'team': h, 'goals': lambda_h, 'cs': np.exp(-lambda_a)})
            fix_preds.append({'event': r['event'], 'team': a, 'goals': lambda_a, 'cs': np.exp(-lambda_h)})

        df_fix = pd.DataFrame(fix_preds)
        past = self.history[self.history['round'] < current_gw]

        # Project player points
        projections = []
        for _, p in self.players.iterrows():
            chance = p.get('chance_of_playing_next_round', 100)
            if pd.isna(chance):
                chance = 100
            if p['status'] in ['s', 'n'] or chance < 25:
                continue

            hist = past[past['player_id'] == p['id']]
            active = hist[hist['minutes'] > 0]
            
            if len(active) < 3:
                continue

            avg_mins = active.tail(5)['minutes'].mean()
            if avg_mins < 30:
                continue

            prob_play = (chance / 100.0) * (0.8 if chance == 75 else 1.0)
            total_mins = hist['minutes'].sum()
            
            g90 = (hist['goals_scored'].sum() / total_mins * 90) if total_mins > 0 else 0
            a90 = (hist['assists'].sum() / total_mins * 90) if total_mins > 0 else 0

            team_fix = df_fix[df_fix['team'] == p['team']]
            total_xp, next_gw_xp = 0, 0
            pos = p['element_type']
            pts = POINTS_MAP[pos]

            for _, f in team_fix.iterrows():
                play_factor = avg_mins / 90.0 * prob_play
                
                xp = f['goals'] * (g90 / 1.3) * play_factor * pts['goal']
                xp += f['goals'] * (a90 / 1.3) * play_factor * pts['assist']
                
                if pos in [1, 2] and avg_mins >= 60:
                    xp += f['cs'] * pts['cs'] * prob_play
                
                xp += (2 if avg_mins >= 60 else 1) * prob_play
                
                total_xp += xp
                if f['event'] == current_gw:
                    next_gw_xp = xp

            projections.append({
                'id': p['id'], 'name': p['web_name'], 'team_id': p['team'],
                'pos_id': pos, 'price': p['now_cost'] / 10.0,
                'horizon_xp': total_xp, 'next_gw_xp': next_gw_xp
            })

        return pd.DataFrame(projections)

    def optimize_squad(self, df, budget=100.0):
        """Select optimal 15-player squad using linear programming."""
        prob = pulp.LpProblem("FPL", pulp.LpMaximize)
        ids = df['id'].tolist()
        x = pulp.LpVariable.dicts("p", ids, cat='Binary')
        
        # Objective: maximize expected points
        prob += pulp.lpSum(df[df['id'] == i]['horizon_xp'].values[0] * x[i] for i in ids)
        
        # Constraints
        prob += pulp.lpSum(df[df['id'] == i]['price'].values[0] * x[i] for i in ids) <= budget
        prob += pulp.lpSum(x[i] for i in ids) == 15
        
        for pos, count in SQUAD_COUNTS.items():
            prob += pulp.lpSum(x[i] for i in ids if df[df['id'] == i]['pos_id'].values[0] == pos) == count

        for t in df['team_id'].unique():
            prob += pulp.lpSum(x[i] for i in ids if df[df['id'] == i]['team_id'].values[0] == t) <= 3

        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        if pulp.LpStatus[prob.status] != 'Optimal':
            return pd.DataFrame()

        selected = [i for i in ids if x[i].varValue == 1]
        return df[df['id'].isin(selected)].copy()

    def pick_team_sheet(self, squad: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
        """
        Select starting XI and bench from squad.
        Returns: (starters, bench, captain_id, vice_captain_id)
        """
        squad = squad.sort_values('next_gw_xp', ascending=False)
        
        starters = pd.concat([
            squad[squad['pos_id'] == pos].head(count)
            for pos, count in STARTING_COUNTS.items()
        ]).sort_values('pos_id')
        
        bench = squad[~squad['id'].isin(starters['id'])].sort_values('next_gw_xp', ascending=False)
        
        top2 = starters.nlargest(2, 'next_gw_xp')
        cap_id, vc_id = top2.iloc[0]['id'], top2.iloc[1]['id']
        
        return starters, bench, cap_id, vc_id

    def format_squad(self, df):
        """Format squad for display with readable names."""
        out = df.copy()
        out['team'] = out['team_id'].map(TEAM_MAP).fillna(out['team_id'])
        out['position'] = out['pos_id'].map(POS_MAP)
        out['horizon_xp'] = out['horizon_xp'].round(1)
        out['next_gw_xp'] = out['next_gw_xp'].round(1)
        return out[['name', 'team', 'position', 'price', 'next_gw_xp', 'horizon_xp']]