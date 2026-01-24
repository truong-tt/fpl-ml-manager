# FPL ML Manager — Overview (25/26)

This project uses Machine Learning to manage an FPL team by predicting player points and optimizing the squad and starting XI. If you’re unfamiliar with FPL, see [FPL_101.md](FPL_101.md).

## How to Check the Lineup

- AI updates the squad weekly via GitHub Actions (Wednesdays).
- View Optimized Squad: see [data/optimal_squad_live.csv](../data/optimal_squad_live.csv) (if present) for player, team, position, and 3-gameweek expected points (XP).
- View Starting XI for the next gameweek: open the repository’s Actions tab, select the latest “FPL Weekly Update” run, and check the build summary for the Starting XI, Captain, and Vice-Captain.

## Data Pipeline & Resilience

- **Flat directory:** All CSV outputs are saved directly in [data](../data) (no `data/raw` vs `data/processed`) to avoid path resolution issues.
- **Type conversion:** API numeric strings are parsed to floats (xG, xA, xGI) via a cleaning step before persistence.
- **Request filtering:** Detailed player history is fetched only for players with minutes > 0 or price > 4.0 to reduce API load.
- **Retries & caching:** API calls use a short retry loop to handle timeouts; local CSV presence skips re-fetch in the main pipeline.

## What the Model Predicts

- **Label:** `total_points` (player points for a specific gameweek)
- **Feature Focus:**
  - **Explosiveness:** Emphasizes chance quality created/received, not just outcomes.
  - **Usage Rate:** Share of team xGI; “Talismans” (>30% team xGI) receive a 1.1× multiplier. See Player Performance Modeling for details.
  - **Minutes & Availability:** Discounts projections for low minute profiles; modeled with appearance probability and role minutes. See Availability & Risk.
  - **Work Rate:** CBIT (clearances, blocks, interceptions, tackles) informs BPS likelihood. See Defensive Usage.
  - **Bayesian Smoothing:** ~400-minute league prior stabilizes per-90 rates. See Bayesian Smoothing.

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

Adapts the NBA “Usage Rate” (percentage of team plays used) to football’s team ecosystem.

- **Finite Pie Theory:** A team has finite output (Goals, xG, Defensive Actions).
- **Usage Calculation:** A player’s historical share of team output determines their slice (e.g., “40% of Man City’s attack flows through Haaland”).
- **Vacuum Effect:** If a high-usage player misses a game, their share is redistributed to teammates rather than disappearing.

### A) Attacking Usage

Weighted mix of Expected Goals (xG) and Expected Assists (xA) defines a player’s attacking slice. We allocate team goal involvement by each player’s share of xGI.

$$
	ext{GoalShare}_{\text{player}} = \frac{\text{xG}_{\text{player}} + \text{xA}_{\text{player}}}{\sum_{p \in \text{TeamXI}} (\text{xG}_p + \text{xA}_p)}
$$

Prediction:

$$
XP_{\text{attack}} = \lambda_{\text{team}} \times \text{GoalShare} \times \text{PlayTimeFactor}
$$

- **Talisman bonus:** The top xGI generator per team/position receives a 1.1× multiplier to their share to reflect on-ball centrality.

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

### D) Bayesian Smoothing

To dampen outliers from small samples, each player’s per-90 rates are combined with a league-average prior equivalent to ~400 minutes:

$$
	ext{Rate}_{\text{shrunk}} = \frac{\text{Rate}_{\text{player}} \cdot M + \text{Rate}_{\text{league}} \cdot 400}{M + 400}
$$

This shrinks extreme values toward the league mean while preserving information from actual minutes $M$.

## Squad Optimization (Linear Programming)

Treat squad selection as a constrained knapsack problem using `pulp`.

### Objective

Maximize total expected points (XP) over 3 gameweeks:

$$
	ext{Maximize } Z = \sum_{i=1}^{N} XP_i \cdot x_i
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

The search returns recommended moves ranked by expected points uplift over the forecast horizon.

## Starting XI and Captaincy

- Requirments: 3+ defenders, 2+ midfielders, 1+ forward.
- Select highest projected scorers to fill Starting XI slots.
- **Captain:** Player with the highest projected points (XP).