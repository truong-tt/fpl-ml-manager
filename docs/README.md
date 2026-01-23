# FPL ML Manager — Overview (25/26)

## How to Check the Lineup

The AI updates the squad every Friday via GitHub Actions.

- **View Squad:** Open `data/processed/optimal_squad_live.csv`. This table contains player names, teams, positions, and expected points (XP).
- **View Starting XI:** Click the **Actions** tab, select the latest **FPL Weekly Update** run, and expand the **Run Optimization** step to see the **1-4-4-2** team sheet and captaincy.

This project predicts Fantasy Premier League points and uses those predictions to help pick an optimal squad and starting XI.

## 1) FPL Scoring (Quick Reference 25/26)

### Minutes Played
- < 60 minutes: +1
- ≥ 60 minutes: +2 (required for clean sheet points)

### Goals (by position)
- Forward: +4 per goal
- Midfielder: +5 per goal
- Defender: +6 per goal
- Goalkeeper: +10 per goal

### Assists and Penalties
- Assist (all positions): +3
- Penalty miss: -2
- Goalkeeper penalty save: +5

### Defensive
- Clean sheet (must play ≥ 60 minutes):
  - GK/DEF: +4
  - MID: +1
- Goals conceded (GK/DEF): -1 per 2 goals conceded
- Own goal (all): -2

### Goalkeeper Saves
- +1 for every 3 saves

### Defensive Contributions (CBIT rule)
- Defenders: +2 for 10+ combined clearances/blocks/interceptions/tackles (CBIT)
- Midfielders/Forwards: +2 for 12+ CBIT + recoveries

### Bonus Points (BPS)
- 1st: +3
- 2nd: +2
- 3rd: +1

### Cards
- Yellow: -1
- Red: -3

## 2) ML Model (What it predicts)

### Label
- `total_points` (player points for a gameweek)

### Feature focus
- Explosiveness: xG, xA
- Stability: probability of playing 60+ minutes, clean sheet likelihood
- Work rate: CBIT thresholds
- Bonus potential: smaller contributions that increase BPS

## 3) Team Strength Model (Poisson Regression)

I model football matches as a stochastic process where the number of goals scored by a team follows a Poisson distribution.

### The Math

For a fixture between Home Team ($i$) and Away Team ($j$), the expected goals ($\lambda$) are:

$$\lambda_{i} = \exp(\alpha_i + \beta_j + \gamma + \delta)$$

$$\lambda_{j} = \exp(\alpha_j + \beta_i + \delta)$$

Where:
- $\alpha$ (Alpha): attack strength of the team
- $\beta$ (Beta): defense weakness of the opponent
- $\gamma$ (Gamma): home advantage constant
- $\delta$ (Delta): league-wide intercept (average goal rate)

### Training (Maximum Likelihood Estimation)

I use `scipy.optimize.minimize` to find the optimal values for $\alpha$ and $\beta$ for every team by minimizing the negative log-likelihood of the observed scores:

$$\mathcal{L}(\theta) = -\sum_{k=1}^{N} \left( y_k \ln(\lambda_k) - \lambda_k \right)$$

By minimizing this loss function, I learn each team’s attacking and defensive strength, adjusted for opponent difficulty.

## 4) Player Performance Model (Risk-Adjusted Expected Value)

Once I have team expected goals ($\lambda$), I project individual player points using a risk-adjusted expected value (EV) calculation.

### A) Base rates (recent form)

I calculate a player's underlying performance metrics per 90 minutes using a weighted historical window (e.g., last 5 active matches):
- $G_{90}$: goals per 90
- $A_{90}$: assists per 90

### B) Fixture scaling

I project a player’s output for a specific fixture by scaling their base rates using the team model’s expected goals:

$$xG_{\text{player}} = \lambda_{\text{team}} \times \left( \frac{G_{90}}{1.3} \right) \times \text{PlayTimeFactor}$$

Note: `1.3` is a calibration factor converting team xG to the approximate sum of individual xG.

### C) Availability & risk (role minutes vs probability)

To avoid the “clean sheet trap” (where lowering minutes removes clean sheet upside), I model minutes in two parts:

- Role minutes ($M_{role}$): expected minutes when the player features (e.g., 90, 80)
- Probability ($P_{\text{play}}$): chance the player appears (from FPL “chance of playing”)

The calculation:

$$EV = P_{\text{play}} \times \left[ (xG \times 4) + (xA \times 3) + (\text{CS}_{\text{points}} \text{ if } M_{role} \ge 60) \right]$$

Why this matters: if a defender has a 50% chance of playing, I calculate points for a full appearance (including clean sheet) and then multiply by 0.5, instead of giving them 45 minutes and removing clean sheet points entirely.

## 5) Squad Optimization (Linear Programming)

I treat squad selection as a knapsack-style optimization solved with `pulp`.

### Objective function

Maximize total expected points ($XP$) over a short horizon (commonly 3 gameweeks):

$$\text{Maximize } Z = \sum_{i=1}^{N} XP_i \cdot x_i$$

Where $x_i$ is a binary decision variable (1 if a player is selected, 0 otherwise).

### Constraints

Budget:

$$\sum_{i=1}^{N} \text{Cost}_i \cdot x_i \le 100.0$$

Squad size:

$$\sum_{i=1}^{N} x_i = 15$$

Max 3 players per club:

$$\sum_{i \in \text{team}=T} x_i \le 3$$

Position structure:

$$\sum GK = 2$$
$$\sum DEF = 5$$
$$\sum MID = 5$$
$$\sum FWD = 3$$

## 6) Starting XI and Captaincy (Team sheet logic)

After selecting the 15, I:
- Enforce a fixed formation (currently 1-4-4-2)
- Pick the top projected scorers for the next gameweek to fill:
  - 1 GK, 4 DEF, 4 MID, 2 FWD
- Captain: highest projected points in the starting XI