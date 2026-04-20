# Autonomous Fantasy Premier League (FPL) ML Manager

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![PyMC](https://img.shields.io/badge/PyMC-Bayesian_Inference-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Tree_Boosting-green.svg)
![PuLP](https://img.shields.io/badge/PuLP-MILP_Optimization-yellow.svg)

## 1. Motivation and Abstract

I built this project to bridge my passion for Premier League football with my interest in Machine Learning and Operations Research. Fantasy Premier League (FPL) is a highly stochastic environment constrained by strict rules (budget, positional limits, team caps) and high-variance outcomes. This makes it a perfect sandbox for combinatorial optimization and predictive modeling.

> **Note:** If you are unfamiliar with the domain-specific rules, scoring systems, and constraints of the game, please read the [FPL 101](FPL_101.md) before exploring the system architecture.

This project implements an autonomous ML agent designed to solve the FPL management problem. The system operates as a data-driven pipeline that:

1. Models match outcomes via **Bayesian inference**.
2. Normalizes player underlying metrics (Opponent-Adjusted xG/xA).
3. Predicts playing time probabilities using **Tree-Based Machine Learning**.
4. Solves a multi-period constrained knapsack problem using **Mixed-Integer Linear Programming (MILP)** to dictate optimal portfolio selection and weekly transfers.

---

## 2. Mathematical Modeling and Machine Learning

### 2.1 Bayesian Poisson Regression via ADVI (Match Simulation)

Matches are modeled as a stochastic process where goals scored follow a Poisson distribution, accounting for bivariate dependence (e.g., 0-0 draws) via a Dixon-Coles adjustment.

Instead of computationally heavy Markov Chain Monte Carlo (MCMC) sampling, this engine utilizes **Automatic Differentiation Variational Inference (ADVI)** via `PyMC` to approximate the posterior distributions of latent team strengths in a fraction of the time, making it viable for standard hardware.

For a fixture between Home ($i$) and Away ($j$), the expected goals ($\lambda$) are defined as:

$$\lambda_{i} = \exp(\alpha_i + \beta_j + \gamma + \delta)$$
$$\lambda_{j} = \exp(\alpha_j + \beta_i + \delta)$$

*Where $\alpha$ is attack strength, $\beta$ is defense weakness, $\gamma$ is home-field advantage, and $\delta$ is the baseline scoring rate.*

**References:**
- [Modeling Association Football Scores and Inefficiencies in the Football Betting Market — Dixon & Coles, 1997](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/1467-9884.00065)
- [Automatic Differentiation Variational Inference — Kucukelbir et al., 2016](https://arxiv.org/abs/1603.00788)

---

### 2.2 The "Form Trap": Opponent-Adjusted Expected Metrics

A common pitfall in sports analytics is over-indexing on players who artificially inflate their underlying numbers against weak opposition. To combat this, the pipeline extracts the Bayesian defensive posterior ($\beta$) of the historical opponent faced, and mathematically divides the player's historical xG/xA by that parameter before feeding it to the time-series decay function. This penalizes stat-padding against weak defenses and rewards production against elite defenses.

---

### 2.3 XGBoost: 3-State Expected Value (EV) Formulation

Playing time is rarely binary. Due to soccer substitution rules, 10-minute cameos heavily penalize FPL scoring. We model playing time as a discrete probability distribution over three states: **Start** (≥ 60 min), **Sub** (< 60 min), and **Bench** (0 min).

The probability of each state is predicted using an **XGBoost Classifier** trained on time-series lagged features (e.g., 3-week rolling average minutes, previous match binary state). The total Expected Value (EV) is the weighted sum of these conditional states:

$$E[\text{Points}] = P(\text{Start}) \cdot E[\text{Pts} \mid \text{Start}] + P(\text{Sub}) \cdot E[\text{Pts} \mid \text{Sub}]$$

**Reference:**
- [XGBoost: A Scalable Tree Boosting System — Chen & Guestrin, 2016](https://arxiv.org/abs/1603.02754)

---

## 3. Combinatorial Optimization (MILP)

Squad selection is framed as a variation of the bounded knapsack problem. We use the `pulp` library (CBC solver) to maximize the Expected Points (XP) over a 5-step time horizon, factoring in risk (variance) inspired by Modern Portfolio Theory (MPT).

### Objective Function

Maximize the risk-adjusted returns of the selected portfolio, while accounting for a captaincy multiplier ($c_i$) and a covariance/stacking bonus ($\omega \cdot y_T$):

$$\max_{x, c, y} \sum_{i=1}^{N} \left( \mu_i - \nu \sigma_i \right)(x_i + c_i) + \sum_{T \in \mathcal{T}} \omega \cdot y_T$$

### Constraints

| # | Constraint | Description |
|---|------------|-------------|
| 1 | **Budget** | $\sum (x_i \cdot \text{Cost}_i) \le \text{Total Bank}$ |
| 2 | **Cardinality** | Exact positional counts: 2 GK, 5 DEF, 5 MID, 3 FWD (15 total) |
| 3 | **Entity Limits** | Max 3 players per Premier League club |
| 4 | **Transaction Costs** | Marginal gain ($\Delta \mu$) must strictly exceed the point penalty for executing a transfer |

---

## 4. Installation and Usage

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

> **Note:** On the first run, the system will automatically pull API data, engineer time-series features, and train the XGBoost minutes classifier before running the optimization sequence.

---

## 5. Future Work

- **Receding Horizon Control (RHC):** Transitioning the multi-period transfer heuristic from static block optimization to true RHC, allowing the time-dimension variable ($x_{i,t}$) to dictate exact transfer execution timing over a rolling window.
- **Deep Learning Integration:** Exploring Recurrent Neural Networks (LSTMs) for time-series forecasting to replace the current Exponentially Weighted Moving Average (EWMA) usage allocations.