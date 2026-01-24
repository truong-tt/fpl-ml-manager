# FPL ML Manager — Overview (25/26)

This project uses Machine Learning to manage an FPL team by predicting player points and optimizing the squad and starting XI. If you’re unfamiliar with FPL, see [docs/FPL_101.md](docs/FPL_101.md).

## How to Check the Lineup

- AI updates the squad weekly via GitHub Actions (Wednesdays).
- View Optimized Squad: open [data/processed/optimal_squad_live.csv](data/processed/optimal_squad_live.csv) for player, team, position, and 3-gameweek expected points (XP).
- View Starting XI for next gameweek: go to the repository’s Actions tab, select the latest “FPL Weekly Update” run, and check the build summary for the 1-4-4-2 Starting XI, Captain, and Vice-Captain.

## What the Model Predicts

- **Label:** `total_points` (player points for a specific gameweek)
- **Feature Focus:**
	- **Explosiveness:** xG (Expected Goals), xA (Expected Assists)
	- **Usage:** Share of team possessions and defensive actions
	- **Stability:** Probability of playing 60+ minutes, clean sheet likelihood
	- **Work Rate:** CBIT thresholds (Clearances, Blocks, Interceptions, Tackles)

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

Adapts the NBA “Usage Rate” (the percentage of team plays used by a player while on the floor) concept to football’s "team ecosystem".

- **Finite Pie Theory:** A team has finite output (Goals, xG, Defensive Actions).
- **Usage Calculation:** A player’s historical share of team output determines their slice (e.g., “40% of Man City’s attack flows through Haaland”).
- **Vacuum Effect:** If a high-usage player misses a game, their share is redistributed to teammates rather than disappearing.

### A) Attacking Usage

Weighted mix of Threat (Shots/Touches) and Creativity (Passes) defines a player’s attacking slice.

$$
	ext{GoalShare}_{\text{player}} = \frac{\text{Threat}_{\text{player}}}{\text{Threat}_{\text{TeamXI}}}
$$

Prediction:

$$
XP_{\text{attack}} = \lambda_{\text{team}} \times \text{GoalShare} \times \text{PlayTimeFactor}
$$

### B) Defensive Usage (CBIT)

Models FPL defensive work points via share of defensive workload.

- **Opponent Pressure:** Scale expected defensive actions based on opponent strength (higher xG → more potential defensive bonus points).
- **Threshold Probability:** Sigmoid function for probability of hitting the 10+ actions threshold (+2 points).

### C) Availability & Risk

Avoids the “clean sheet trap” by modeling minutes in two parts:

- **Role Minutes ($M_{role}$):** Expected minutes when the player features.
- **Probability ($P_{\text{play}}$):** Chance the player appears.

Overall expected value:

$$
EV = P_{\text{play}} \times \left[ XP_{\text{attack}} + XP_{\text{CBIT}} + (\text{CS}_{\text{points}} \text{ if } M_{role} \ge 60) \right]
$$

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

## Starting XI and Captaincy

- Enforce a 1-4-4-2 formation for the immediate gameweek.
- Select highest projected scorers to fill Starting XI slots.
- **Captain:** Player with the highest projected points (XP).