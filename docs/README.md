# Autonomous Fantasy Premier League (FPL) ML Manager

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Tree_Boosting-green.svg)
![PuLP](https://img.shields.io/badge/PuLP-MILP_Optimization-yellow.svg)
![GitHub Actions](https://img.shields.io/github/actions/workflow/status/your-username/fpl-ml-manager/weekly_update.yml?label=Weekly%20Bot&logo=github)

## 1. Motivation and Abstract

I built this project to bridge my passion for Premier League football with my interest in Machine Learning and Operations Research. Fantasy Premier League (FPL) is a highly stochastic environment constrained by strict rules (budget, positional limits, team caps) and high-variance outcomes. This makes it a perfect sandbox for combinatorial optimization and predictive modeling.

> **Note:** If you are unfamiliar with the domain-specific rules, scoring systems, and constraints of the game, please read the [FPL 101 Primer](FPL_101.md) before exploring the system architecture.

This project implements an autonomous ML agent designed to solve the FPL management problem. The system operates as a data-driven pipeline that:

1. Models match outcomes via **XGBoost Poisson Regression**.
2. Normalizes player underlying metrics using a **two-way opponent adjustment** (attacking metrics via opponent xGA; defensive action metrics via opponent xG).
3. Predicts playing time probabilities using **Tree-Based Machine Learning**.
4. Solves a multi-period constrained knapsack problem using **Mixed-Integer Linear Programming (MILP)** with **Receding Horizon Control (RHC)** to dictate optimal portfolio selection and weekly transfers.

---

## 2. Mathematical Modeling and Machine Learning

### 2.1 XGBoost Poisson Regression (Match Simulation)

The previous Bayesian Hierarchical Poisson model (PyMC/ADVI) has been replaced with a **Gradient Boosting Poisson Regression** approach. The core reason for this change is data scope: the FPL API only exposes single-season data, which means the sparse dataset does not provide enough signal for a hierarchical Bayesian model to reliably estimate latent team-strength posteriors. On such datasets, XGBoost's ability to capture non-linear tactical interactions — such as a high-press attack meeting a deep defensive block — generalizes better than the previous EWMA-smoothed Poisson regression.

Two independent XGBoost models are trained using a `count:poisson` objective, one for each side of the fixture. Each model outputs the expected goal rate ($\lambda$) directly:

$$\lambda_{\text{home}} = f_{\text{home}}(\mathbf{x}_{\text{home}}, \mathbf{x}_{\text{away}})$$

$$\lambda_{\text{away}} = f_{\text{away}}(\mathbf{x}_{\text{away}}, \mathbf{x}_{\text{home}})$$

Where $\mathbf{x}$ is the engineered feature vector for each team (see Section 2.2). The trained models are exported as `xgb_home_goals.json` and `xgb_away_goals.json` via `src/train_match_model.py`.

**Reference:**
- [XGBoost: A Scalable Tree Boosting System — Chen & Guestrin, 2016](https://arxiv.org/abs/1603.02754)

---

### 2.2 Feature Engineering (`src/features.py`)

A dedicated feature engineering module generates rolling time-series metrics as inputs to the XGBoost match models. The feature set per fixture includes the rolling 5-game xG and rolling 5-game xGA for both the home and away team, as well as a strength differential term capturing the relative attacking and defensive quality between the two sides. These rolling windows smooth out single-match variance while remaining responsive to recent tactical form.

---

### 2.3 FPL Scoring Model

The point valuation layer has been corrected and extended to more accurately reflect the official FPL scoring rules.

**Goalkeeper Goals** — corrected from 6 points to the accurate value of **10 points**.

**Defensive Action Bonuses** — Poisson simulations are run over historical defensive action rates to compute the probability of a player triggering the bonus threshold in a given match. Importantly, those historical action counts are first opponent-adjusted using the attacking quality scalar described in Section 2.4, so the simulation operates on quality-normalized rates rather than raw totals. The bonus is position-dependent:

- Defenders receive **+2 points** for achieving $\geq 10$ combined Clearances, Blocks, Interceptions, and Tackles (CBIT).
- Midfielders and Forwards receive **+2 points** for achieving $\geq 12$ combined Clearances, Blocks, Interceptions, Tackles, and Recoveries (CBIRT).

**Negative Point Deductions** — expected deductions are now factored into each player's EV calculation using historical per-90 rates for yellow cards ($-1$), red cards ($-3$), missed penalties ($-2$), and own goals ($-2$).

---

### 2.4 The "Form Trap": Opponent-Adjusted Expected Metrics

A common pitfall in sports analytics is over-indexing on players who artificially inflate their underlying numbers against weak opposition. To combat this, the pipeline applies a **two-way opponent adjustment** before feeding data into the time-series decay function:

- **Attacking Metrics (xG/xA):** Adjusted using a defensive quality scalar derived from the historical opponent's rolling xGA. This penalizes attackers who stat-pad against weak defenses and rewards those who produce against elite ones.
- **Defensive Bonus Actions (CBIT/CBIRT):** Adjusted using an attacking quality scalar derived from the historical opponent's rolling xG. This penalizes defenders who accumulate high clearance and tackle numbers simply because they were pinned back by a heavily dominant attacking side.

This dual-scalar approach ensures the model rewards true underlying quality rather than situational stat-padding — capturing the "Form Trap" symmetrically for both attackers and defenders.

---

### 2.5 XGBoost: 3-State Expected Value (EV) Formulation

Playing time is rarely binary. Due to soccer substitution rules, 10-minute cameos heavily penalize FPL scoring. We model playing time as a discrete probability distribution over three states: Start ($\geq 60$ min), Sub ($< 60$ min), and Bench ($0$ min).

The probability of each state is predicted using an **XGBoost Classifier** trained on time-series lagged features (e.g., 3-week rolling average minutes, previous match binary state). The total Expected Value (EV) is the weighted sum of these conditional states:

$$E[\text{Points}] = P(\text{Start}) \cdot E[\text{Pts} \mid \text{Start}] + P(\text{Sub}) \cdot E[\text{Pts} \mid \text{Sub}]$$

**Reference:**
- [XGBoost: A Scalable Tree Boosting System — Chen & Guestrin, 2016](https://arxiv.org/abs/1603.02754)

---

## 3. Combinatorial Optimization (MILP) and Receding Horizon Control

Squad selection is framed as a variation of the bounded knapsack problem. We use the `pulp` library (CBC solver) to maximize the Expected Points (XP) over a 5-step time horizon, factoring in risk (variance) inspired by Modern Portfolio Theory (MPT).

Because FPL enforces strict transaction costs (free transfers can be banked, excess transfers cost -4 points), the optimizer uses **Receding Horizon Control (RHC)**. Rather than a static 1D block optimization, the decision variable is expanded to a 2D matrix ($x_{i,t}$), allowing the time-dimension to dictate exact transfer execution timing over a rolling window.

### Objective Function

Maximize the risk-adjusted returns of the selected portfolio across the horizon $T$, while accounting for transfer point penalties ($h_t$):

$$\max_{x,\, c,\, h} \sum_{t=1}^{T} \sum_{i=1}^{N} \left( \mu_{i,t} - \nu \sigma_i \right)(x_{i,t} + c_{i,t}) - \sum_{t=1}^{T} 4 h_t$$

### Constraints

| # | Constraint | Description |
|---|------------|-------------|
| 1 | **Dynamic Budget** | $\sum_{i} x_{i,t} \cdot \text{Cost}_{i,t} \leq \text{Bank}_t$ for every gameweek $t$ |
| 2 | **Cardinality** | Exact positional counts: 2 GK, 5 DEF, 5 MID, 3 FWD (15 total) |
| 3 | **Entity Limits** | Max 3 players per Premier League club |
| 4 | **Conservation of Transfers** | Linking constraints track accumulated Free Transfers (capped at 5) and trigger the $-4$ point penalty only when $h_t > 0$ |

---

## 4. Project Structure

```
fpl-ml-manager/
├── src/
│   ├── main.py                  # Orchestrator: runs the full pipeline end-to-end
│   ├── data_loader.py           # Fetches and sanitizes data from the FPL public API
│   ├── fpl_engine.py            # Core FPLEngine class: XGBoost predictions + PuLP MILP solver
│   ├── features.py              # Rolling time-series feature engineering
│   ├── train_match_model.py     # Trains and serializes the XGBoost match (goals) models
│   └── train_minutes_model.py   # Trains and serializes the XGBoost minutes classifier
├── data/
│   ├── fixtures.csv             # Raw fixture data from the FPL API
│   ├── history.csv              # Player gameweek history
│   ├── players.csv              # Player metadata and stats
│   ├── teams.csv                # Team metadata
│   ├── xgb_home_goals.json      # Serialized XGBoost home goals model
│   ├── xgb_away_goals.json      # Serialized XGBoost away goals model
│   └── processed/
│       ├── lineup_output.txt    # Human-readable weekly squad summary
│       └── optimal_squad_live.csv  # Raw optimized squad data (updated each run)
├── docs/
│   ├── README.md                # This file
│   └── FPL_101.md               # Primer on FPL rules and scoring for new readers
└── .github/
    └── workflows/
        └── weekly_update.yml    # GitHub Actions workflow (runs every Wednesday 10:00 UTC)
```

---

## 5. Automated Deployment (GitHub Actions)

The bot is fully self-contained and runs on a schedule via **GitHub Actions** — no manual intervention is needed during the season. Every **Wednesday at 10:00 UTC**, the `weekly_update.yml` workflow triggers automatically. It executes `src/main.py`, prints the weekly squad summary to the GitHub Step Summary UI, and commits the updated `data/` artifacts (including `optimal_squad_live.csv`) back to the repository.

This means the repository itself always reflects the bot's latest recommended squad after each weekly run.

---

## 6. Installation and Usage

### Prerequisites

- Python 3.11+
- Git

### Local Setup

**1. Clone the repository:**

```bash
git clone https://github.com/your-username/fpl-ml-manager.git
cd fpl-ml-manager
```

**2. Install the required dependencies:**

```bash
pip install -r requirements.txt
```

**3. Run the autonomous manager:**

```bash
python src/main.py
```

> **Note:** On the first run, the system will automatically pull API data from the public FPL API (no authentication required), extract and sanitize the required columns (`recoveries`, `yellow_cards`, `red_cards`, `penalties_missed`, `own_goals`), engineer time-series features via `src/features.py`, and train both the XGBoost match simulation models and the minutes classifier. If the match model files (`xgb_home_goals.json`, `xgb_away_goals.json`) are not found on disk, `src/main.py` will automatically trigger `src/train_match_model.py` before running the optimization sequence.

### Output

After a successful run, the pipeline produces three outputs:

- **Console** — the weekly summary is printed directly, including the Starting XI, Substitutes, Captain, Vice-Captain, and recommended transfers.
- `data/processed/lineup_output.txt` — the same summary saved as a plain-text file.
- `data/processed/optimal_squad_live.csv` — the full optimized squad as a structured CSV for downstream analysis.

---

## 7. Future Work

- **Deep Learning Integration:** Exploring Recurrent Neural Networks (LSTMs) for time-series forecasting to replace the Exponentially Weighted Moving Average (EWMA) usage allocations.
- **Live Price-Change Inference:** Integrating market movement predictors to execute MILP transactions prior to player cost fluctuations, maximizing team value over a 38-week season.