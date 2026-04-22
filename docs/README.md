# Autonomous Fantasy Premier League ML Manager

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-quantile_%2B_poisson-green.svg)
![PuLP](https://img.shields.io/badge/PuLP-MILP%20%2B%20RHC-yellow.svg)

A data-driven agent that picks and manages a 15-player FPL squad end-to-end. The pipeline learns per-player point distributions with quantile-regression gradient boosting, estimates match-level goal rates with a Dixon-Coles-corrected Poisson model, and then solves a single joint MILP for the 15-man squad, the starting XI, and the captain across a rolling horizon.

> New to FPL scoring rules and constraints? Start with the [FPL 101 primer](FPL_101.md) — the math below assumes you understand clean sheets, appearance points, and the transfer/chip system.

---

## 1. System Architecture

```
FPL API  ──►  data_loader  ──►  players.csv, teams.csv, fixtures.csv, history.csv
                                          │
                                          ▼
                                     features.py
                        Elo snapshots + rolling team/player metrics
                                          │
                 ┌────────────────────────┼────────────────────────┐
                 ▼                                                 ▼
        train_match_model                                 train_points_model
     2 × Poisson XGBoost + DC τ                        3 × Quantile XGBoost (q10/q50/q90)
                 │                                                 │
                 └────────────────────────┬────────────────────────┘
                                          ▼
                                     fpl_engine
                     Per-(player, GW) projection frame with xp_t and var_t
                                          │
                                          ▼
                                     optimizer
             MILP: squad × XI × captain + RHC transfer/hit accounting
                                          │
                                          ▼
                                       chips
                           Greedy TC / BB / FH / WC scheduler
                                          │
                                          ▼
                                   lineup.md + snapshot
```

All model artifacts (Poisson boosters, quantile boosters) and intermediate CSVs are persisted to `data/` so subsequent runs only retrain when an artifact is missing.

---

## 2. Feature Engineering (`src/features.py`)

### 2.1 Team Elo

Team strengths are replayed chronologically over all finished fixtures using a standard Elo update with home-field advantage and a margin-of-victory (MoV) multiplier:

$$
E_h = \frac{1}{1 + 10^{-(R_h + \text{HFA} - R_a)/400}}, \qquad S_h = \begin{cases} 1 & g_h > g_a \\ 0.5 & g_h = g_a \\ 0 & g_h < g_a \end{cases}
$$

$$
R_h' = R_h + K \cdot \text{MoV} \cdot (S_h - E_h), \qquad \text{MoV} = (|g_h - g_a| + 1)^{0.4}
$$

The MoV exponent follows FiveThirtyEight's sports-Elo methodology [[6](#ref-538)]; it dampens the effect of blowouts while still rewarding decisive wins. For every fixture (including upcoming), we stamp the **pre-match** Elo of both sides (`elo_h_pre`, `elo_a_pre`) so it can be used as a leakage-free feature.

Hyperparameters: $K = 20$, $\text{HFA} = 60$, initial rating $= 1500$.

### 2.2 Rolling team and player metrics

For the match model, we aggregate player histories into per-(team, GW) sums of $\{xG, xGA, GF, GA\}$ and roll them forward at three windows $w \in \{3, 5, 10\}$, always **shifted by one GW** so training features for fixture at GW $t$ only use information available before GW $t$:

$$
\overline{xG}_{T,t}^{(w)} = \frac{1}{w} \sum_{k=t-w}^{t-1} xG_{T,k}
$$

For the points model, each player row uses lagged minutes ($m_{i,t-1}, m_{i,t-2}, m_{i,t-3}$), 5- and 10-GW rolling per-player means of underlying metrics $\{xG, xA, xGI, BPS, ICT, \text{saves}, CBI, \text{tackles}, \text{recoveries}, \text{pts}\}$, and context pulled from the fixture join: `is_home`, `opp_xg_5`, `opp_xga_5`, `opp_elo`, `own_elo`, and `elo_gap`. Set-piece and penalty-taker flags come directly from the FPL API.

---

## 3. Match Model (`src/train_match_model.py`)

### 3.1 Poisson goal rates

Two independent XGBoost regressors with a `count:poisson` objective model expected goals for each side [[1](#ref-xgboost)]:

$$
\lambda_h = f_h(\mathbf{x}), \qquad \lambda_a = f_a(\mathbf{x})
$$

where $\mathbf{x}$ is the 31-dimensional match feature vector from §2. A Bayesian hierarchical Poisson (à la Maher [[3](#ref-maher)]) was rejected because the FPL API only exposes a single season of data; with $\lesssim 380$ finished fixtures, partial-pooling posteriors cannot out-perform a tree ensemble that captures non-linear interactions (press-intensity × block-depth, Elo-gap × home-advantage, etc.).

### 3.2 Dixon-Coles low-score correction

Independent Poisson marginals over-predict $(0,1)$ and $(1,0)$ and under-predict $(0,0)$ and $(1,1)$ in football [[2](#ref-dc)]. Dixon & Coles fix this with a multiplicative correction $\tau$ applied to the joint PMF on the four low-scoring corners:

$$
P(X = x, Y = y) = \tau_{\lambda_h, \lambda_a}(x, y) \cdot \frac{\lambda_h^x e^{-\lambda_h}}{x!} \cdot \frac{\lambda_a^y e^{-\lambda_a}}{y!}
$$

$$
\tau(x, y) = \begin{cases}
1 - \lambda_h \lambda_a \rho & (x, y) = (0, 0) \\
1 + \lambda_h \rho & (x, y) = (0, 1) \\
1 + \lambda_a \rho & (x, y) = (1, 0) \\
1 - \rho & (x, y) = (1, 1) \\
1 & \text{otherwise}
\end{cases}
$$

We fix $\rho = -0.10$ (negative, as in the original paper; see note in [[2](#ref-dc)] §4). The truncated joint PMF is normalized to sum to 1 over $\{0, \dots, 8\}^2$.

### 3.3 Clean-sheet probabilities (analytic)

Unlike the previous iteration which Monte-Carlo'd clean sheets and inverted the correction, here CS is computed directly from the DC joint matrix $M$:

$$
P(\text{home CS}) = \sum_{x=0}^{8} M_{x, 0}, \qquad P(\text{away CS}) = \sum_{y=0}^{8} M_{0, y}
$$

Sanity check: holding $\lambda_h = 1.5$ fixed, increasing $\lambda_a$ from $0.5$ to $2.5$ drops $P(\text{home CS})$ from $0.61$ to $0.08$. The sign is now correct.

---

## 4. Player Points Model (`src/train_points_model.py`)

### 4.1 Why quantile regression

FPL points per player per GW are discrete, heavy-tailed, and bimodal (zero for DNPs + a wide distribution when playing). A single point estimate of the mean throws away information the optimizer needs — specifically, we want the **ceiling** (for Triple Captain timing), a **downside floor** (for risk budgeting), and a consistent **variance estimate** (for portfolio selection). We therefore train three independent XGBoost regressors using the `reg:quantileerror` objective with $\alpha \in \{0.10, 0.50, 0.90\}$, minimizing the pinball loss [[4](#ref-koenker)]:

$$
\mathcal{L}_\alpha(y, \hat{y}) = \begin{cases}
\alpha (y - \hat{y}) & y \geq \hat{y} \\
(1 - \alpha)(\hat{y} - y) & y < \hat{y}
\end{cases}
$$

Target: raw FPL `total_points` per player per GW. **This subsumes every scoring rule end-to-end.** The model learns goal points, assist points, clean-sheet bonuses, the defensive-action bonus thresholds, BPS, and all negative deductions jointly from the data — there is no hand-coded scoring table.

### 4.2 Post-hoc non-crossing

Independently fit quantile regressors can cross (pathological $\hat{q}_{10} > \hat{q}_{50}$). We enforce monotonicity row-wise by sorting the three predictions ascending at inference:

$$
[\hat{q}_{10}, \hat{q}_{50}, \hat{q}_{90}] \leftarrow \text{sort}([\hat{q}_{10}, \hat{q}_{50}, \hat{q}_{90}])
$$

Simple and arguably less principled than constrained optimization [[5](#ref-chernozhukov)], but empirically affects $< 2\%$ of rows in our data.

### 4.3 Inference: handling DGWs, BGWs, and injuries

For each player $i$ and upcoming GW $t$, we locate every fixture their club plays that GW (0, 1, or 2) and build one feature row per fixture. Quantile predictions are then:

1. **Scaled by forward-looking availability** $a_i = \text{chance of playing next round}_i / 100$, zeroed if the player's `status` is suspended/out/unavailable. Historical availability is already captured implicitly through the lagged-minute features, so this only adds forward-looking injury information.
2. **Aggregated across fixtures per $(i, t)$** by simple summation:

$$
\hat{q}^{(i,t)}_\alpha = \sum_{f \in F_{i,t}} a_i \cdot \hat{q}^{(i,f)}_\alpha, \quad \alpha \in \{0.10, 0.50, 0.90\}
$$

A blank GW yields $F_{i,t} = \emptyset$ and therefore $\hat{q}^{(i,t)}_\alpha = 0$; a double GW stacks both fixtures additively.

### 4.4 Variance estimate from quantile spread

The optimizer's risk term needs a scalar variance per $(i, t)$. Assuming the points distribution is approximately Gaussian in the central mass (a coarse but serviceable approximation for players expected to play), the $[q_{10}, q_{90}]$ range spans $\Phi^{-1}(0.9) - \Phi^{-1}(0.1) \approx 2.56$ standard deviations:

$$
\hat{\sigma}^2_{i,t} \approx \left( \frac{\hat{q}^{(i,t)}_{90} - \hat{q}^{(i,t)}_{10}}{2.56} \right)^2
$$

This is lighter than a full Monte-Carlo covariance estimate (which can't be consumed by the linear CBC MILP solver anyway) but preserves the core signal: players with wide quantile spreads are penalized more heavily in the squad objective.

---

## 5. Combinatorial Optimization (`src/optimizer.py`)

### 5.1 Decision variables

Per player $i \in \{1, \dots, N\}$ and GW $t \in \{t_0, \dots, t_0 + H - 1\}$ (horizon $H = 5$):

| Variable | Domain | Meaning |
|---|---|---|
| $x_{i,t}$ | $\{0, 1\}$ | In 15-man squad at GW $t$ |
| $s_{i,t}$ | $\{0, 1\}$ | In starting XI at GW $t$ |
| $c_{i,t}$ | $\{0, 1\}$ | Captain at GW $t$ |
| $\text{tin}_{i,t}$ | $\{0, 1\}$ | Newly transferred IN at GW $t$ |
| $\text{ft}_t$ | $\{1, \dots, 5\}$ | Free transfers banked entering GW $t$ |
| $\text{sv}_t$ | $\{0, \dots, 5\}$ | Free transfers saved out of GW $t$ |
| $h_t$ | $\mathbb{Z}_{\geq 0}$ | Number of 4-pt hits taken at GW $t$ |

For the cold-start solve, $x_{i,t}$ collapses to a single $x_i$ (no transfers yet — squad fixed across the horizon).

### 5.2 Objective

$$
\max \sum_{t=t_0}^{t_0 + H - 1} \sum_{i=1}^{N} \Big[ \underbrace{\mu_{i,t} s_{i,t}}_{\text{starter XP}} + \underbrace{b \cdot \mu_{i,t} (x_{i,t} - s_{i,t})}_{\text{bench auto-sub}} + \underbrace{\mu_{i,t} c_{i,t}}_{\text{captain double}} - \underbrace{\nu \hat{\sigma}^2_{i,t} x_{i,t}}_{\text{risk}} + \underbrace{\eta \mu_{i,t} (1 - \text{EO}_i) x_{i,t}}_{\text{differential tilt}} \Big] - \underbrace{\sum_{t} 4 h_t}_{\text{hits}}
$$

Design notes:

- $\mu_{i,t} = \hat{q}^{(i,t)}_{50}$ is the median EV — more robust to the heavy right tail than the mean for this objective.
- The captain term adds $\mu_{i,t}$ on top of the starter term, which already counts $\mu_{i,t}$ once — summing to the correct doubling. Triple Captain is handled separately in the chip module (§7).
- Bench weight $b = 0.15$ is an empirical estimate of auto-sub realization (roughly: $P(\text{bench player auto-subbed in}) \times$ average fraction of starter's points retained).
- **EO tilt** $\eta \cdot \mu (1 - \text{EO})$ is zero by default, which targets pure points EV. Setting $\eta > 0$ late in the season pushes the MILP toward differentials (high EV, low ownership) to maximize rank-EV rather than points-EV. This is a linearization of the classic "rank chase" objective; Markowitz-style portfolio theory [[7](#ref-markowitz)] motivates the $\mu - \nu \sigma^2$ structure of the baseline term.
- The quadratic portfolio variance $x^\top \Sigma x$ would require MIQP; since CBC is LP-only, we take the diagonal approximation. Within-team correlation is bounded by the 3-per-club constraint and implicit in learned $\hat{\sigma}^2_{i,t}$ values anyway.

### 5.3 Structural constraints (applied at every $t$)

Squad size, positional quotas, club cap, budget:

$$
\sum_i x_{i,t} = 15 \qquad \sum_{i: \text{pos}(i) = p} x_{i,t} = q_p, \quad q = \{1\!\!:\!2,\ 2\!\!:\!5,\ 3\!\!:\!5,\ 4\!\!:\!3\}
$$

$$
\sum_{i: \text{club}(i) = k} x_{i,t} \leq 3 \quad \forall k \in \text{clubs} \qquad \sum_i p_i x_{i,t} \leq B_t
$$

where $B_t = B_0 + \text{bank}$ (effective budget equals previous squad value plus uninvested bank).

Starting XI and captain:

$$
\sum_i s_{i,t} = 11 \qquad \sum_i c_{i,t} = 1 \qquad c_{i,t} \leq s_{i,t} \leq x_{i,t}
$$

$$
\sum_{i: \text{pos}(i) = 1} s_{i,t} = 1 \qquad \sum_{i: \text{pos}(i) = p} s_{i,t} \geq r_p, \quad r = \{1\!\!:\!1,\ 2\!\!:\!3,\ 3\!\!:\!2,\ 4\!\!:\!1\}
$$

The $c_{i,t} \leq s_{i,t}$ constraint closes the captain-on-the-bench bug present in the previous iteration — the captain is now guaranteed to start.

### 5.4 Transfer accounting (RHC only)

Transfer indicator for player $i$ at GW $t$:

$$
\text{tin}_{i,t} \geq x_{i,t} - x_{i,t-1}, \qquad \text{tin}_{i,t} \in \{0, 1\}
$$

with $x_{i, t_0 - 1} = \mathbf{1}[i \in \text{prior squad}]$. Free-transfer conservation, with a cap of 5:

$$
\text{ft}_t = \begin{cases} \text{ft}^{\text{init}} & t = t_0 \\ 1 + \text{sv}_{t-1} & t > t_0 \end{cases}, \qquad \text{sv}_t \in \{0, \dots, 5\}
$$

Transfer budget at each GW — total transfers in equal free transfers used plus hits:

$$
\sum_i \text{tin}_{i,t} = (\text{ft}_t - \text{sv}_t) + h_t, \qquad h_t \geq 0
$$

Combined with the $-4 h_t$ term in the objective, the solver only commits a hit when expected gain strictly exceeds 4 points.

### 5.5 Receding Horizon Control

The full multi-period MILP is solved with $H = 5$ look-ahead each week, but only the decisions for $t_0$ (the next GW) are executed — `transfers_in`, `transfers_out`, `xi_ids`, `captain`, `vice`, `hits`. The following week's run re-solves from the updated state $(\text{squad}_{t_0}, \text{bank}_{t_0}, \text{ft}_{t_0 + 1})$. This is a standard Receding Horizon Control (model-predictive control) formulation [[8](#ref-rhc)], which balances long-horizon optimality against the wrongness of far-future EV predictions.

---

## 6. Chip Scheduling (`src/chips.py`)

Chip activation is a convex function of fixture quality that the weekly MILP does not see directly, so it's handled as a greedy post-processing heuristic over the same projection frame:

| Chip | Decision rule |
|---|---|
| **Triple Captain** | $\arg\max_t \max_{i \in \text{squad}} \hat{q}^{(i,t)}_{50}$ |
| **Bench Boost** | $\arg\max_t \sum_{i \in \text{bench}} \hat{q}^{(i,t)}_{50}$ |
| **Free Hit** | $\arg\max_t |\text{teams blanking at } t|$ within horizon |
| **Wildcard** | Trigger if RHC proposes $\geq 4$ transfers in or $\geq 2$ hits |

The wildcard rule leverages the MILP's own willingness to pay hits as a signal that the current squad is far from optimal.

---

## 7. Project Structure

```
fpl-ml-manager/
├── src/
│   ├── main.py                  # Orchestrator + markdown report writer
│   ├── data_loader.py           # FPL API fetcher
│   ├── features.py              # Elo + rolling team/player features
│   ├── train_match_model.py     # Poisson goals + DC τ + analytic CS
│   ├── train_points_model.py    # Quantile XGBoost (q10/q50/q90)
│   ├── fpl_engine.py            # Inference engine: projection frame builder
│   ├── optimizer.py             # MILP squad + XI + captain, RHC transfers
│   └── chips.py                 # TC / BB / FH / WC heuristics
├── data/
│   ├── players.csv, teams.csv, fixtures.csv, history.csv
│   ├── xgb_home_goals.json, xgb_away_goals.json
│   ├── xgb_points_q10.json, xgb_points_q50.json, xgb_points_q90.json
│   └── processed/
│       ├── lineup.md            # Weekly markdown report
│       └── squad_snapshot.csv   # Carried state for next RHC pass
├── docs/
│   ├── README.md                # This file
│   └── FPL_101.md               # Domain primer
└── .github/workflows/
    └── weekly_update.yml        # GitHub Actions: Wednesdays 10:00 UTC
```

---

## 8. Installation and Usage

Requires Python 3.11+.

```bash
git clone https://github.com/truong-tt/fpl-ml-manager
cd fpl-ml-manager
pip install -r requirements.txt
python src/main.py
```

First run trains every model artifact from scratch; subsequent runs reuse `data/*.json` and only retrain when a file is missing. Output is written to `data/processed/lineup.md`.

---

## 9. References

<a id="ref-xgboost"></a>**[1]** Chen, T. and Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. Proceedings of KDD '16. [Link](https://arxiv.org/abs/1603.02754).

<a id="ref-dc"></a>**[2]** Dixon, M.J. and Coles, S.G. (1997). *Modelling Association Football Scores and Inefficiencies in the Football Betting Market*. Journal of the Royal Statistical Society: Series C (Applied Statistics), Vol. 46, No. 2. [Link](https://www.ajbuckeconbikesail.net/wkpapers/Airports/MVPoisson/soccer_betting.pdf).

<a id="ref-maher"></a>**[3]** Maher, M.J. (1982). *Modelling Association Football Scores*. Statistica Neerlandica, Vol. 36, Issue 3. [Link](http://www.90minut.pl/misc/maher.pdf).

<a id="ref-koenker"></a>**[4]** Koenker, R. and Bassett, G. (1978). *Regression Quantiles*. Econometrica, Vol. 46, No. 1. [Link](https://people.eecs.berkeley.edu/~jordan/sail/readings/koenker-bassett.pdf).

<a id="ref-chernozhukov"></a>**[5]** Chernozhukov, V., Fernández-Val, I. and Galichon, A. (2010). *Quantile and Probability Curves Without Crossing*. Econometrica, Vol. 78, No. 3. [Link](http://alfredgalichon.com/wp-content/uploads/2012/10/Econometrica_article_may-2010.pdf).

<a id="ref-538"></a>**[6]** Silver, N. and Fischer-Baum, R. *How We Calculate NBA Elo Ratings*. FiveThirtyEight methodology post (2015) — source for the MoV exponent convention used here. [Link](https://fivethirtyeight.com/features/how-we-calculate-nba-elo-ratings/).

<a id="ref-markowitz"></a>**[7]** Markowitz, H. (1952). *Portfolio Selection*. Journal of Finance, Vol. 7, No. 1. [Link](http://efinance.org.cn/cn/fm/Portfolio%20Selection.pdf). Motivates the $\mu - \nu \sigma^2$ structure of the squad objective.

<a id="ref-rhc"></a>**[8]** Mayne, D.Q., Rawlings, J.B., Rao, C.V. and Scokaert, P.O.M. (2000). *Constrained Model Predictive Control: Stability and Optimality*. Automatica, Vol. 36, Issue 6. [Link](https://www.researchgate.net/profile/Saeed-Rahmati-2/post/Dual-mode-versus-Min-MaxLMI-Based-MPC/attachment/5bf0f01d3843b00675457f08/AS%3A694179780890625%401542516765597/download/constrained+model+predictive+control+stability+and+optimality+%28automatica2000%29.pdf). Canonical reference for Receding Horizon Control.

<a id="ref-elo"></a>**[9]** Elo, A.E. (1978). *The Rating of Chessplayers, Past and Present*. Arco Publishing. [Link](https://gwern.net/doc/statistics/order/comparison/1978-elo-theratingofchessplayerspastandpresent.pdf) Original Elo rating system — formulas in §2.1 derive directly from this.

<a id="ref-pulp"></a>**[10]** Mitchell, S., O'Sullivan, M. and Dunning, I. (2011). *PuLP: A Linear Programming Toolkit for Python*. University of Auckland tech report. The CBC solver bundled with PuLP is from the COIN-OR project [Link](https://github.com/coin-or/Cbc).

<a id="ref-fpl"></a>**[11]** Premier League. *Fantasy Premier League Public API*. Undocumented but stable at `https://fantasy.premierleague.com/api/`. Endpoints used: `bootstrap-static/`, `fixtures/`, `element-summary/{id}/`.

---

## 10. Future Work

- **Correlated-risk portfolio objective.** Replace the diagonal $\hat{\sigma}^2$ penalty with a proper $x^\top \Sigma x$ term driven by joint match-player Monte Carlo. Requires migrating from CBC to a MIQP solver (Gurobi / CPLEX / SCIP).
- **Price-change prediction.** Model overnight price deltas from `transfers_in_event` / `transfers_out_event` to time transfers before price moves, preserving squad value over the season.
- **Set-piece / manager changes.** Embedding-based detection of regime changes (new manager, new set-piece taker) that invalidate historical rolling features.
- **Learned chip scheduler.** Re-formulate chip activation as a jointly-solved MILP extension rather than a post-hoc heuristic.