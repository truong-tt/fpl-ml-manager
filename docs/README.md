# FPL ML Manager — Overview (25/26)

This project uses Machine Learning to manage an FPL team by predicting player points and optimizing the squad and starting XI. If you’re unfamiliar with FPL, see [FPL_101.md](FPL_101.md).

## How to Check the Lineup

- AI updates the squad weekly via GitHub Actions (Wednesdays).
- View Optimized Squad: see [data/optimal_squad_live.csv](../data/optimal_squad_live.csv) (if present) for player, team, position, and 5-gameweek expected points (XP).
- View Starting XI for the next gameweek: open the repository’s Actions tab, select the latest “FPL Weekly Update” run, and check the build summary for the Starting XI, Captain, and Vice-Captain.

## What the Model Predicts

- **Label:** `total_points` (player points for a specific gameweek)
- **Feature Focus:**
  - **Recency-Weighted Form:** Uses only the **last 7 matches** with time-decay weights `[1.0, 1.0, 0.8, 0.6, 0.4, 0.3, 0.2]` — the 2 most recent matches weighted equally highest, then declining.
  - **5-Gameweek Horizon:** Projects expected points (`horizon_xp`) over the next 5 gameweeks for squad optimization.
  - **FPL Team Strength Metrics:** Incorporates `strength`, `strength_overall_home`, and `strength_overall_away` from the official FPL API.
  - **Explosiveness:** Emphasizes chance quality created/received, not just outcomes.
  - **Usage Rate:** Share of team xGI within the 7-match window. See Player Performance Modeling for details.
  - **Minutes & Availability:** Discounts projections for low minute profiles; modeled with appearance probability and role minutes. See Availability & Risk.
  - **Work Rate:** CBIT (clearances, blocks, interceptions, tackles) informs BPS likelihood. See Defensive Usage.

## Team Strength Modeling (Poisson Regression)

Matches are modeled as a stochastic process where goals follow a Poisson distribution.

### The Math

For a fixture between Home Team ($i$) and Away Team ($j$), expected goals ($\lambda$):

$$
\lambda_{i} = \exp(\alpha_i + \beta_j + \gamma + \delta)
$$
$$
\lambda_{j} = \exp(\alpha_j + \beta_i + \delta)
$$

**Parameters**
- $\alpha$ (Alpha): Attack strength of the team
- $\beta$ (Beta): Defense weakness of the opponent
- $\gamma$ (Gamma): Home advantage constant
- $\delta$ (Delta): League-wide intercept (average goal rate); without this, the model might think 2-1 is equal to 4-2.

### Training (Maximum Likelihood Estimation)

Optimize $\alpha$ and $\beta$ per team using `scipy.optimize.minimize`, minimizing the negative log-likelihood of observed scores:

$$
\mathcal{L}(\theta) = -\sum_{k=1}^{N} \left( y_k \ln(\lambda_k) - \lambda_k \right)
$$

This learns each team’s attacking and defensive strength, adjusted for opponent difficulty.

## Player Performance Modeling (Usage Rate & Risk)

Adapts the NBA "Usage Rate" (percentage of team plays used) to football's team ecosystem. **All player stats are calculated from the last 7 matches** with the same recency weights as team strength.

- **Finite Pie Theory:** A team has finite output (Goals, xG, Defensive Actions).
- **Usage Calculation:** A player's share of team output over the last 7 matches determines their slice (e.g., "40% of Man City's attack flows through Haaland").
- **Recency-Weighted Stats:** Recent performances (GW-1, GW-2) count more than older ones.

### A) Attacking Usage

Weighted mix of Expected Goals (xG) and Expected Assists (xA) over the 7-match window defines a player's attacking slice:

$$
\text{GoalShare}_{\text{player}} = \frac{\sum_{gw} w_{gw} \cdot (\text{xG}_{gw} + \text{xA}_{gw})}{\sum_{p \in \text{Team}} \sum_{gw} w_{gw} \cdot (\text{xG}_{p,gw} + \text{xA}_{p,gw})}
$$

Prediction:

$$
XP_{\text{attack}} = \lambda_{\text{team}} \times \text{GoalShare} \times \text{PlayTimeFactor}
$$

- **Share caps:** Goal share capped at 70%, assist share at 50% to prevent over-reliance on single players.

### B) Defensive Usage (CBIT)

Models FPL defensive work points via share of defensive workload.

- **Opponent Pressure:** Scale expected defensive actions based on opponent strength (higher xG → more potential defensive bonus points).
- **Threshold Probability:** Sigmoid function for probability of hitting the 10+ actions threshold (+2 points).

### C) Availability & Risk

Avoids the “clean sheet trap” by modeling minutes in two parts:

- **Role Minutes ($M_{\text{role}}$):** Expected minutes when the player features.
- **Probability ($P_{\text{play}}$):** Chance the player appears.

Overall expected value:

$$
EV = P_{\text{play}} \times \left[ XP_{\text{attack}} + XP_{\text{CBIT}} + (\text{CS}_{\text{points}} \text{ if } M_{\text{role}} \ge 60) \right]
$$

### D) Nailedness Factor

Players are assigned a "nailedness" multiplier based on their minutes share over the last 7 matches:

| Minutes Share | Multiplier |
|--------------|------------|
| ≥90%         | 1.05       |
| ≥75%         | 1.00       |
| ≥50%         | 0.85       |
| <50%         | 0.50       |

This rewards consistent starters and penalizes rotation risks.

## Squad Optimization (Linear Programming)

Treat squad selection as a constrained knapsack problem using `pulp`.

### Objective

Maximize total expected points (XP) over **5 gameweeks** (`horizon_xp`):

$$
\text{Maximize } Z = \sum_{i=1}^{N} XP_i \cdot x_i
$$

Where $x_i$ is binary (1 if selected, 0 otherwise).

### Constraints

- **Budget:** $\sum \text{Cost}_i \cdot x_i \le 100.0$
- **Squad Size:** $\sum x_i = 15$
- **Club Limit:** For any team $T$, $\sum_{i \in \text{team}=T} x_i \le 3$
- **Structure:** 2 GK, 5 DEF, 5 MID, 3 FWD

## Transfer Recommendations

The engine simulates near-term scenarios using a beam search over transfer actions:

- **Hold:** No transfer; evaluate current squad vs projections.
- **1 Transfer:** Single optimal move under budget and team constraints.
- **Hit:** Two moves with a -4 point penalty; included when the projected gain compensates.

The search returns recommended moves ranked by expected points uplift over the **5-gameweek horizon**.

## Starting XI and Captaincy

- Requirments: 3+ defenders, 2+ midfielders, 1+ forward.
- Select highest projected scorers to fill Starting XI slots.
- **Captain:** Player with the highest projected points (XP).