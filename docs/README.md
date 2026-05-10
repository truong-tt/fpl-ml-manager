# Autonomous Fantasy Premier League ML Manager

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-quantile_%2B_poisson-green.svg)](https://xgboost.readthedocs.io/)
[![PuLP](https://img.shields.io/badge/PuLP-MILP%20%2B%20RHC-yellow.svg)](https://coin-or.github.io/pulp/)
[![Schedule](https://img.shields.io/badge/GitHub_Actions-2%C3%97daily-lightgrey.svg)](../.github/workflows/pipeline.yml)

Data-driven agent. Pick + manage 15-player Fantasy Premier League squad end-to-end. Weekly pipeline: learn per-position per-player point distributions via quantile-regression gradient boosting; estimate match-level goal rates via Dixon–Coles-corrected Poisson; solve one joint MILP for squad + XI + captain over rolling horizon; schedule chips on top. Data: [olbauday/FPL-Core-Insights](https://github.com/olbauday/FPL-Core-Insights) CSVs (FPL API + Opta per-match stats + ClubElo, refreshed 2×/day) + thin live-FPL-API overlay for current-GW prices + injury status.

> **New to FPL?** Start with [FPL 101 primer](FPL_101.md). Math below assumes clean sheets, appearance points, bench-order auto-sub rule, transfer/chip system.

> **Scope:** Research project, personal use. Past model performance no guarantee future rank — league noisy, luck-driven by design.

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Feature Engineering](#2-feature-engineering)
3. [Match Model](#3-match-model)
4. [Player Points Model](#4-player-points-model)
5. [Combinatorial Optimization](#5-combinatorial-optimization)
6. [Chip Scheduling](#6-chip-scheduling)
7. [Validation and Calibration](#7-validation-and-calibration)
8. [Repository Layout](#8-repository-layout)
9. [Installation and Usage](#9-installation-and-usage)
10. [Recently Shipped](#10-recently-shipped)
11. [Future Work](#11-future-work)
12. [References](#12-references)
13. [Data and Credit](#data-and-credit)
14. [License](#license)

---

## 1. System Architecture

End-to-end pipeline. FPL-Core-Insights CSVs + live-FPL-API price overlay → markdown report: squad, XI, captain, transfers, hits, chip recs. Refreshed 2×/day on upstream data update.

```mermaid
flowchart TD
    A["FPL-Core-Insights CSVs + live FPL API overlay<br/>teams, players, fixtures (Opta + ClubElo), playermatchstats, player_gameweek_stats"]
    B["1. Data Loader<br/>src/data_loader.py<br/>players, teams, fixtures, history CSVs"]
    C["2. Feature Engineering<br/>src/features.py<br/>ClubElo + EMA team/player metrics + per-player Opta"]
    D["3a. Match Model<br/>src/train_match_model.py<br/>2× Poisson XGBoost + Dixon-Coles τ"]
    E["3b. Points Model<br/>src/train_points_model.py<br/>4 positions × 3 quantiles = 12 XGBoost regressors"]
    M["3c. Minutes Model<br/>src/train_minutes_model.py<br/>two-stage P(plays) × E[mins/90 | plays]"]
    N["3d. Bonus Head<br/>src/train_bonus_model.py<br/>3-quantile XGBoost on FPL bonus column"]
    F["4. Inference Engine<br/>src/fpl_engine.py<br/>joint MC aggregation: μ (Swanson), s (std), cap_xp"]
    G["5. MILP Optimizer<br/>src/optimizer.py<br/>squad × XI × captain over horizon H = 8 (RHC + attenuation)"]
    H["6. Chip Scheduler<br/>src/chips.py<br/>greedy TC / BB / FH / WC heuristics"]
    I["lineup.md + squad_snapshot.csv"]
    A --> B --> C
    C --> D
    C --> E
    C --> M
    C --> N
    D --> F
    E --> F
    M --> F
    N --> F
    F --> G --> H --> I
```

Model artifacts (Poisson + quantile boosters) + intermediate CSVs persist under `data/`. Subsequent runs retrain only on missing artifacts. GitHub Actions workflow [.github/workflows/pipeline.yml](../.github/workflows/pipeline.yml) re-runs pipeline twice daily — 05:30 + 17:30 UTC, 30 min after FPL-Core-Insights upstream refresh windows. Walk-forward recalibration (points + minutes heads) auto-fires when recalib JSONs older than `RECALIB_STALE_DAYS` (= 14) — see `_maybe_recalibrate` in [src/main.py](../src/main.py).

---

## 2. Feature Engineering

Lives in [src/features.py](../src/features.py). Two feature families: team-level state for match model; player-level lags for points model. Both use strict GW-shift to prevent leakage.

### 2.1 Team Elo

Pre-match Elo per side precomputed on every fixture from FPL-Core-Insights (ClubElo, point-in-time per match — see [clubelo.com](https://clubelo.com)). Stamped on fixture as `elo_h_pre` / `elo_a_pre`. Chronological replay below = fallback when dataset missing/null:

$$E_h = \frac{1}{1 + 10^{-(R_h + \text{HFA} - R_a)/400}}$$

$$
S_h = \begin{cases}
1   & \text{if } g_h > g_a \\
0.5 & \text{if } g_h = g_a \\
0   & \text{if } g_h < g_a
\end{cases}
$$

$$R_h' = R_h + K \cdot \text{MoV} \cdot (S_h - E_h), \qquad \text{MoV} = (\lvert g_h - g_a \rvert + 1)^{0.4}$$

MoV-multiplier from FiveThirtyEight sports-Elo — see [How We Calculate NBA Elo Ratings][ref-538]. Exponent $0.4$ differs from 538's NBA formula but same family; dampens blowouts, rewards decisive wins. Elo chess → football well-studied — see [Using ELO ratings for match result prediction in association football][ref-hvattum] for bookmaker-odds validation. Hyperparameters follow community sports-Elo conventions.

| Hyperparameter (fallback only) | Value |
|---|---|
| K-factor | 20 |
| Home-field advantage (HFA) | 60 Elo |
| Initial rating | 1500 |

### 2.2 Rolling team and player metrics

**Match** model: two stat families rolled at $w \in \{3, 5, 10\}$ for FPL block, $w = 5$ for Opta block. Always **shift-1** — features for fixture at GW $t$ use only info available before GW $t$. Aggregator = exponentially weighted moving average with **half-life** $w/2$, not equal-weight rolling mean — recent GWs decay slowly, older GWs fade exponentially, no hard cutoff at $w$:

$$\overline{\text{xG}}_{T,t}^{(w)} = \frac{\sum_{k=1}^{t-1} \alpha_k \cdot \text{xG}_{T,k}}{\sum_{k=1}^{t-1} \alpha_k}, \qquad \alpha_k = (1/2)^{(t - 1 - k) / (w/2)}$$

| Source | Stats |
|---|---|
| Aggregated FPL `history` (per-team, per-GW sums) | xG, xGA, GF, GA |
| Opta team-level on `fixtures.csv` (one row per side per match) | true Opta xG (`oxg`), big chances (`obc`), total shots (`osh`), and each conceded counterpart |

**Stakes** features (both models, [src/league_table.py](../src/league_table.py)): per-(season, event) league standings reconstructed from finished fixtures. Per side, signed pts gap to each tier cutoff normalised by max remaining pts:

$$\Delta_k = \frac{P_k - P_\text{self}}{3 \cdot G_\text{rem}}, \qquad k \in \{1, 4, 6, 17\}$$

for title / UCL (top 4) / European cups band (top 6, UEL + UECL) / safety (17th). Match model gets `h_*` + `a_*` cols; points model side-conditions to `own_*` / `opp_*`. Late-season step-changes (Arsenal title chase, mid-table beach mode, drop-fight desperation) propagate immediately rather than bleeding through the EMA halflife window. Encodes a structurally novel signal — booster cannot infer "Arsenal needs 3 more wins to clinch" from rolling xG alone.

**Points** model: per-player row uses lagged minutes $m_{i,t-1}, m_{i,t-2}, m_{i,t-3}$ + 5-/10-GW rolling per-player means of (xG, xA, xGI, BPS, ICT, saves, CBI, tackles, recoveries) + six per-player Opta stats from `playermatchstats.csv` (aggregated to per-(player, GW) sums first): Opta xG `oxg`, Opta xA `oxa`, chances created `occ`, touches in opp box `otob`, total shots `osh`, dribbles `odrib`. Fixture context from match-feature join: `is_home`, `opp_xg_5`, `opp_xga_5`, `opp_elo`, `own_elo`, `elo_gap`. Set-piece + pen-taker flags from FPL playerstats. Rolling `total_points` excluded on purpose — feedback loop: premium with one bad recent GW projects lower forever. Underlying xG/xA/ICT carry form signal without that pathology.

---

## 3. Match Model

Lives in [src/train_match_model.py](../src/train_match_model.py).

### 3.1 Poisson goal rates

Two independent XGBoost regressors with `count:poisson` objective model expected goals per side. Backbone: [XGBoost: A Scalable Tree Boosting System][ref-xgboost]:

$$\lambda_h = f_h(\mathbf{x}), \qquad \lambda_a = f_a(\mathbf{x})$$

where $\mathbf{x}$ = 31-dim match feature vector from §2. Bayesian hierarchical Poisson — see [Bayesian hierarchical model for the prediction of football results][ref-baio] — rejected: FPL API exposes single season, ~380 fixtures. Partial-pooling posteriors can't beat tree ensemble capturing non-linear interactions (press × block depth, Elo gap × home advantage, etc).

### 3.2 Dixon–Coles low-score correction

Independent Poisson marginals over-predict $(0,1)$ + $(1,0)$, under-predict $(0,0)$ + $(1,1)$ in football. [Modelling Association Football Scores and Inefficiencies in the Football Betting Market][ref-dc] fixes via multiplicative correction $\tau$ on joint PMF at four low-scoring corners:

$$P(X = x, Y = y) = \tau_{\lambda_h, \lambda_a}(x, y) \cdot \frac{\lambda_h^x e^{-\lambda_h}}{x!} \cdot \frac{\lambda_a^y e^{-\lambda_a}}{y!}$$

$$
\tau(x, y) = \begin{cases}
1 - \lambda_h \lambda_a \rho & \text{if } (x, y) = (0, 0) \\
1 + \lambda_h \rho           & \text{if } (x, y) = (0, 1) \\
1 + \lambda_a \rho           & \text{if } (x, y) = (1, 0) \\
1 - \rho                     & \text{if } (x, y) = (1, 1) \\
1                            & \text{otherwise}
\end{cases}
$$

Default $\rho = -0.10$ (negative, per original paper). Truncated joint PMF normalized to sum to 1 over $\{0, \dots, 8\}^2$.

ρ tunable, but **only joint score distribution identifiable** — τ correction redistributes mass between (0,0)↔(0,1) and (1,0)↔(1,1), conserves row/column marginals exactly, so clean-sheet probabilities (column/row sums) mathematically invariant to ρ. Prior `cs_brier`-based grid = no-op. [src/tune_dc_rho.py](../src/tune_dc_rho.py) now grid-searches $\rho \in \{-0.20, -0.15, -0.10, -0.05, 0\}$ (configurable), ranks by mean negative log-likelihood of **actual scoreline** under DC-corrected joint PMF. Output → `data/processed/backtest/dc_rho_grid.csv`.

### 3.3 Clean-sheet probabilities (analytic)

Clean sheets computed directly from Dixon–Coles joint matrix $M$, not Monte Carlo:

$$P(\text{home CS}) = \sum_{x=0}^{8} M_{x, 0}, \qquad P(\text{away CS}) = \sum_{y=0}^{8} M_{0, y}$$

Sanity check: hold $\lambda_h = 1.5$ fixed, raise $\lambda_a$ from $0.5$ to $2.5$ → $P(\text{home CS})$ drops $0.61 \to 0.08$. Sign correct, response monotone.

---

## 4. Player Points Model

Lives in [src/train_points_model.py](../src/train_points_model.py).

### 4.1 Why quantile regression

FPL points per player per GW: discrete, heavy-tailed, bimodal (zero on DNP + wide distribution when playing). Single mean estimate throws away info optimizer needs:

| Quantile | Use in optimizer |
|---|---|
| q10 (downside floor) | Risk budgeting, spread estimate |
| q50 (central mass) | Median anchor — combined with q10 / q90 into Swanson mean (§4.4) for the optimizer's μ |
| q90 (ceiling) | Triple Captain timing, spread estimate |

Train **per-position** XGBoost regressors — one per (position $\times$ quantile) cell, $4 \times 3 = 12$ boosters total — `reg:quantileerror` objective, $\alpha \in \{0.10, 0.50, 0.90\}$. Directly minimizes pinball loss from [Regression Quantiles][ref-koenker]:

$$
\mathcal{L}_\alpha(y, \hat{y}) = \begin{cases}
\alpha \cdot (y - \hat{y})       & \text{if } y \geq \hat{y} \\
(1 - \alpha) \cdot (\hat{y} - y) & \text{if } y < \hat{y}
\end{cases}
$$

Target: raw FPL `total_points` per player per GW. **Subsumes every scoring rule end-to-end.** Model learns goal points, assist points, CS bonuses, def-action bonus thresholds, BPS, all negative deductions jointly from data. No hand-coded scoring table.

Single shared model previously regressed premium FWDs toward mid-tier population mean — scoring distributions differ structurally per position (GK saves, DEF CS bonuses, MID score+assist, FWD convert). Position one-hots alone insufficient at this data scale. Per-position models break population-mean trap. Trade-off: each subset small (~3k rows for FWDs), q90 booster outlier-sensitive — fix via strong regularization (`max_depth=3`, `min_child_weight=30`, `reg_alpha=0.5`, `reg_lambda=2.0`) + sanity ceiling clip at 25 (credible single-GW boom: hat-trick + assist + bonus).

### 4.2 Post-hoc non-crossing

Independently fit quantile regressors can cross — predicted 10th percentile > predicted 50th = nonsensical. Enforce row-wise monotonicity by sorting predictions ascending at inference:

$$[\hat{q}_{10},\; \hat{q}_{50},\; \hat{q}_{90}] \;\leftarrow\; \text{sort}\bigl([\hat{q}_{10},\; \hat{q}_{50},\; \hat{q}_{90}]\bigr)$$

Simple. Less principled than constrained optimization à la [Quantile and Probability Curves Without Crossing][ref-chernozhukov], but empirically affects <2% of rows here — not worth the cost for now.

### 4.3 Inference: handling DGWs, BGWs, and injuries

Per player $i$ + upcoming GW $t$: locate every fixture their club plays that GW (0, 1, or 2), build one feature row per fixture. Quantile predictions:

1. **Scaled by expected availability $a_{i,f}$** per fixture. Source: two-stage minutes head ([src/train_minutes_model.py](../src/train_minutes_model.py)). Legacy single `reg:logistic` regressor blurred bimodal target (zero spike for DNP vs ~0.95 mass when playing); replacement trains two heads on disjoint loss surfaces — `binary:logistic` $P(\text{plays})$ on full row set + `reg:logistic` $E[\text{mins}/90 \mid \text{plays}=1]$ on played-only subset. Combined availability $a_{i,f} = P(\text{plays}) \cdot E[\text{mins}/90 \mid \text{plays}=1]$. Engine multiplies $a_{i,f}$ onto $\hat{q}_{10}$ / $\hat{q}_{50}$ (mean-mass requires actual minutes on pitch), multiplies bare $P(\text{plays})$ onto $\hat{q}_{90}$ — given player gets on, ceiling near-fully realised, further discounting by $E[\text{mins} \mid \text{plays}] \approx 0.85$ would systematically under-state haul probability for nailed-but-subbed picks. DGW totals clipped at 1 — playing both legs ≠ "2× available". For immediate next GW only, FPL's `chance_of_playing_next_round / 100` taken as hard upper bound (FPL knows specific injuries model can't infer from history); statuses `s` / `n` / `u` (suspended / not available / unavailable) zero the row.
2. **Aggregated across fixtures per (i, t)** by simple summation:

$$\hat{q}^{(i,t)}_\alpha = \sum_{f \in F_{i,t}} a_i \cdot \hat{q}^{(i,f)}_\alpha, \qquad \alpha \in \{0.10, 0.50, 0.90\}$$

Blank GW: $F_{i,t} = \emptyset$ → $\hat{q}^{(i,t)}_\alpha = 0$. Double GW stacks both fixtures additively.

### 4.4 Swanson mean and dispersion

Optimizer needs scalar EV $\mu_{i,t}$ + scalar dispersion $\hat{s}_{i,t}$ per $(i, t)$. FPL points right-skewed (most weeks 1–2 pts, hauls 8–15 pts) so median $\hat{q}_{50}$ systematically under-shoots mean — summing medians across XI produced ~35-pt totals vs realistic 50–65-pt expectations. Use Swanson / Keefer–Bodily 3-quantile estimator:

$$\mu_{i,t} \;=\; 0.3\,\hat{q}^{(i,t)}_{10} \;+\; 0.4\,\hat{q}^{(i,t)}_{50} \;+\; 0.3\,\hat{q}^{(i,t)}_{90}$$

Earlier revision used Simpson weights $(1, 4, 1)/6$ (mislabeled as "Pearson–Tukey"). Heavy median weight under-fired tail mean on right-skewed FPL points; Swanson 0.3/0.4/0.3 calibrated to lognormal-family distributions, lifts $\mu$ ~12 % on premium archetypes without disturbing rank order. Original Pearson–Tukey (1965) uses $0.185\,q_{05} + 0.63\,q_{50} + 0.185\,q_{95}$ — needs q05/q95 boosters that pinball-loss noise prohibits at 3k rows/position; Swanson is the canonical q10/q90 substitute.

For dispersion, assume central mass approximately Gaussian for players expected to play → q10–q90 spans ~2.56 std devs ($\Phi^{-1}(0.9) - \Phi^{-1}(0.1) \approx 2.56$). Optimizer applies penalty linearly (CBC LP-only, quadratic term needs MIQP), so engine emits **standard deviation**, not its square:

$$\hat{s}_{i,t} \;\approx\; \frac{\hat{q}^{(i,t)}_{90} \;-\; \hat{q}^{(i,t)}_{10}}{2.56}$$

Linear penalty `−λ·s` keeps risk term magnitude comparable to EV term, stops solver dodging high-ceiling players that `s²` term would over-punish. Variance/covariance MIQP upgrade queued for §11.

### 4.5 Captaincy score

Captaincy = separate decision from XI selection. Optimizer captain term contributes $\kappa_{i,t} \cdot c_{i,t}$ independently of $\mu_{i,t} \cdot s_{i,t}$. Pure $\hat{q}_{90}$ as captain reward over-weighted ceiling, crowned low-mean/high-variance players over high-mean MIDs with comparable upside. Anchor on Swanson mean from §4.4, add fraction of upside premium:

$$\kappa_{i,t} \;=\; \mu_{i,t} \;+\; \gamma \cdot \bigl(\hat{q}^{(i,t)}_{90} \;-\; \mu_{i,t}\bigr), \qquad \gamma = 0.3$$

Mean = dominant signal. Ceiling = tiebreaker among similar-mean candidates. `CAP_UPSIDE_WEIGHT` in [src/fpl_engine.py](../src/fpl_engine.py) = tunable knob — lower (e.g. 0.2) for safer captains, raise (0.5+) for boom-chasing.

### 4.6 Isotonic quantile recalibration

Walk-forward backtest (§7) shows raw boosters systematically over-predict central mass + under-shoot right tail on played-only rows. Per-(position, quantile) monotone non-parametric map applied at inference closes gap. Implementation in [src/recalibrate_points.py](../src/recalibrate_points.py):

1. Equal-frequency bin played-only `q_pred` into ~20 buckets per (pos, α) cell.
2. Per bin, take empirical α-quantile of `y` as calibration target (`bin_center → bin_alpha_quantile`).
3. Fit `sklearn.isotonic.IsotonicRegression(out_of_bounds="clip")` on (bin_center, bin_alpha_quantile). Knot pairs serialized as `{type: "iso", knots: [[x_i, y_i], …]}`.

Affine fallback `a + b · q_pred`, $b \in [0.1, 5.0]$, kicks in for any (pos, α) cell with rows < `MIN_ROWS_ISOTONIC` (= 400) — isotonic over-fits per-bin noise on small subsets. Slope floor $b \geq 0.1$ preserves booster's per-row ranking that captaincy + risk terms depend on. Stored as `{type: "affine", ab: [a, b]}`. Legacy `[a, b]` list form auto-detected for backward compat.

JSON schema:

```
{pos_id: {q10|q50|q90: {type: "iso"|"affine", knots|ab: ...}}}
```

Auto-loaded by `train_points_model.predict_quantiles` after row-sort + before sanity clip. Non-crossing re-enforced after recalibration. Same family as §4.7 minutes head.

Held-out validation (early affine variant, GW 32–35 fit on GW 28–30, played-only):

| Level | Coverage (raw) | Coverage (recalib) | Pinball Δ |
| --- | --- | --- | --- |
| q10 | 0.04 | 0.08 | −10% |
| q50 | 0.28 | 0.44 | −7% |
| q90 | 0.79 | 0.86 | −6% |

Isotonic variant supersedes affine on cells with adequate row count; expect comparable or improved pinball deltas pending fresh holdout publication.

### 4.7 Minutes-head isotonic recalibration

§7.4 audit shows raw minutes/90 booster systematically under-predicts playing time at every reliability bucket — predicted played-rate ≈ 0.24 vs actual ≈ 0.34 in most recent holdout. Per-position monotone isotonic map closes gap without disturbing rank order:

$$\hat{p}^{\text{cal}}_i = f_{\text{pos}(i)}\bigl(\hat{p}_i\bigr), \qquad f_{p} \text{ monotone non-decreasing}, \qquad f_{p}(0) = 0,\; f_{p}(1) = 1$$

Fit on walk-forward `minutes_pred.csv` per pos_id ∈ {1, 2, 3, 4} via `sklearn.isotonic.IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)` ([src/recalibrate_minutes.py](../src/recalibrate_minutes.py)). Target = binary `played` (1 if `mins_actual > 0`); raw `mins_pred` treated as $P(\text{played})$. Knot pairs $(x_i, y_i)$ serialized to `data/minutes_recalib.json`.

Why isotonic not affine: minutes head produces ~10× rows per position vs points head (every player every GW vs played-only), so non-parametric fit well-supported. Reliability bins in §7.4 show non-monotone gaps a 2-parameter affine map cannot close.

Inference: `train_minutes_model.predict_minutes(..., apply_recalib=True)` auto-loads JSON, applies linear-interp between knots row-wise before output multiplied onto q10/q50/q90 quantiles in [src/fpl_engine.py](../src/fpl_engine.py). FPL's `chance_of_playing_next_round` hard upper bound for immediate next GW untouched.

### 4.8 Bonus head (additive blend)

Lives in [src/train_bonus_model.py](../src/train_bonus_model.py). Bonus (FPL `bonus` $\in \{0, 1, 2, 3\}$, awarded to top-3 BPS scorers per match) = highest-variance fragment of `total_points`. Main points head sees rolling BPS as feature, learns part of contribution implicitly, but signal diluted across other targets baked into `total_points` (goals, assists, CS, BPS-driven bonus, deductions). Result: flat $\hat{q}_{90}$ ceiling on bonus-heavy archetypes — CS-and-block defender lands near population mean.

Separate quantile booster trained directly on `bonus` preserves discrete 0/1/2/3 mass + asymmetric tail. Three boosters (`reg:quantileerror`, $\alpha \in \{0.10, 0.50, 0.90\}$), single shared model (bonus distribution sparse — splitting into per-position heads shrinks each subset below regularisation budget). Engine sums onto points-head quantiles with damping factor:

$$\hat{q}^{\text{combined}}_\alpha = \hat{q}^{\text{points}}_\alpha + \beta \cdot \hat{q}^{\text{bonus}}_\alpha, \qquad \beta = 0.5$$

Knob `BONUS_BLEND` in [src/fpl_engine.py](../src/fpl_engine.py). $\beta = 1.0$ double-counts: points head already partly learned bonus from `roll5_bps`. $\beta = 0.5$ pragmatic damp pending clean retrain of points head on `total_points - bonus` (Future Work).

### 4.9 Joint Monte-Carlo aggregation

Lives in [src/fpl_engine.py](../src/fpl_engine.py) (`_joint_mc_aggregate`). Prior aggregation summed quantiles across DGW fixtures independently per player + treated different players' uncertainties as uncorrelated — Liverpool clean sheet modelled as raising Virgil's distribution + Salah's distribution as if unrelated, even though both flow from same underlying match outcome. Joint MC step keeps per-row Swanson moments, draws correlated samples:

$$\widetilde{\text{pts}}_{i,f}^{(d)} \;=\; \mu_{i,f} \;+\; \rho \cdot \hat{s}_{i,f} \cdot \eta_{\text{team}(i),\, t(f)}^{(d)} \;+\; \sqrt{1 - \rho^2} \cdot \hat{s}_{i,f} \cdot \varepsilon_{i,f}^{(d)}$$

with $\eta_{k,t}^{(d)} \sim \mathcal{N}(0, 1)$ shared across every player on team $k$ in GW $t$ for draw $d$, $\varepsilon_{i,f}^{(d)} \sim \mathcal{N}(0, 1)$ idiosyncratic per row. `MC_TEAM_RHO` $= 0.4$, `MC_SAMPLES` $= 800$. Aggregation per $(i, t)$ sums $\widetilde{\text{pts}}_{i,f}^{(d)}$ across DGW fixtures within each draw, then takes sample mean ($\mu_{i,t}$), sample std ($\hat{s}_{i,t}$), sample 90-th quantile (input to `cap_xp`).

$\rho = 0$ collapses to prior diagonal aggregation. Non-zero $\rho$ captures within-club covariance directly: Salah+Virgil-rise-together on Liverpool CS, Salah+Diaz-rise-together on goal blitz. Aggregation outputs same `xp_t / var_t / cap_xp_t` schema optimizer in §5 already consumes.

---

## 5. Combinatorial Optimization

Lives in [src/optimizer.py](../src/optimizer.py). Solved with PuLP — see [PuLP: A Linear Programming Toolkit for Python][ref-pulp] — bundled COIN-OR CBC solver.

### 5.1 Decision variables

Per player $i \in \{1, \dots, N\}$ + GW $t \in \{t_0, \dots, t_0 + H - 1\}$, horizon $H = 8$:

| Variable | Domain | Meaning |
|---|---|---|
| $x_{i,t}$ | binary | In 15-man squad at GW $t$ |
| $s_{i,t}$ | binary | In starting XI at GW $t$ |
| $c_{i,t}$ | binary | Captain at GW $t$ |
| $\text{tin}_{i,t}$ | binary | Newly transferred IN at GW $t$ |
| $\text{ft}_t$ | integer, 1 to 5 | Free transfers banked entering GW $t$ |
| $\text{sv}_t$ | integer, 0 to 5 | Free transfers saved out of GW $t$ |
| $h_t$ | non-negative integer | Number of 4-pt hits taken at GW $t$ |

Cold-start solve: $x_{i,t}$ collapses to single $x_i$ (no transfers yet, squad fixed across horizon).

### 5.2 Objective

$$\max \sum_{t=t_0}^{t_0 + H - 1} \sum_{i=1}^{N} \Bigl[\, \mu_{i,t}\, s_{i,t} \;+\; b \cdot \mu_{i,t}\, (x_{i,t} - s_{i,t}) \;+\; \kappa_{i,t}\, c_{i,t} \;-\; \nu\, \hat{s}_{i,t}\, x_{i,t} \;+\; \eta\, \mu_{i,t}\, (1 - \text{EO}_i)\, x_{i,t} \,\Bigr] \;-\; \sum_{t} 4\, h_t$$

| Term | Role |
|---|---|
| $\mu_{i,t}\, s_{i,t}$ | Starter expected points |
| $b \cdot \mu_{i,t}\, (x_{i,t} - s_{i,t})$ | Bench auto-sub EV ($b = 0.15$) |
| $\kappa_{i,t}\, c_{i,t}$ | Captain reward $\kappa_{i,t}$ from §4.5 |
| $-\nu\, \hat{s}_{i,t}\, x_{i,t}$ | Risk penalty (linear, diagonal) |
| $\eta\, \mu_{i,t}\, (1 - \text{EO}_i)\, x_{i,t}$ | Differential / EO tilt (zero by default) |
| $-4\, h_t$ | Hit cost |

Design notes:

- $\mu_{i,t}$ = Swanson mean $0.3\,q_{10} + 0.4\,q_{50} + 0.3\,q_{90}$ from §4.4 — skew-aware estimator, lifts mean ~12 % vs Simpson weights on right-skewed FPL points.
- Bench weight $b = 0.15$ = empirical auto-sub realization, roughly $P(\text{bench player auto-subbed in})$ × avg fraction of starter points retained.
- **Risk penalty** = **linear in standard deviation** $\hat{s}_{i,t}$, not variance. CBC LP-only — quadratic $\hat{s}^2$ term would need MIQP — linear form keeps risk magnitude on same scale as EV term, so solver doesn't over-punish high-ceiling players ($\hat{s}^2$ penalty would scale super-linearly with spread). Reference precedent for Markowitz-style mean–variance trade-off in fantasy sports lineup ILPs: [Picking Winners in Daily Fantasy Sports Using Integer Programming][ref-hvz].
- **EO tilt** $\eta \cdot \mu \cdot (1 - \text{EO})$ defaults zero → pure points EV. $\eta > 0$ late in season pushes solver toward differentials (high EV, low ownership) → max rank-EV.
- Full quadratic portfolio variance $x^\top \Sigma x$ needs MIQP. CBC LP-only → diagonal linear approximation here. Within-team correlation bounded by 3-per-club cap, partly absorbed into learned $\hat{s}_{i,t}$ values.

### 5.3 Structural constraints (applied at every $t$)

Squad size + positional quotas: 2 GK, 5 DEF, 5 MID, 3 FWD = 15:

$$\sum_i x_{i,t} = 15, \qquad \sum_{i\,:\,\text{pos}(i) = p} x_{i,t} = q_p$$

with $q_1 = 2$, $q_2 = 5$, $q_3 = 5$, $q_4 = 3$. Club cap of 3 players per club + budget cap:

$$\sum_{i\,:\,\text{club}(i) = k} x_{i,t} \leq 3 \quad \forall k, \qquad \sum_i p_i\, x_{i,t} \leq B_t$$

$B_t$ = previous squad value + uninvested bank. Starting XI + captain:

$$\sum_i s_{i,t} = 11, \qquad \sum_i c_{i,t} = 1, \qquad c_{i,t} \leq s_{i,t} \leq x_{i,t}, \qquad c_{i,t} = 0 \;\; \forall\, i \text{ s.t. } \text{pos}(i) \in \{1, 2\}$$

Trailing term enforces **MID/FWD-only captaincy rule** — defender booms (CS + goal) correlated with team performance → doubling leaks rank-EV. Uncorrelated upside lives in attack. Formation minima — exactly 1 GK starts, ≥3 DEF, ≥2 MID, ≥1 FWD:

$$\sum_{i\,:\,\text{pos}(i) = 1} s_{i,t} = 1, \qquad \sum_{i\,:\,\text{pos}(i) = p} s_{i,t} \geq r_p$$

$r_1 = 1$, $r_2 = 3$, $r_3 = 2$, $r_4 = 1$. $c_{i,t} \leq s_{i,t}$ guarantees captain not on bench.

### 5.4 Transfer accounting

Transfer indicator for player $i$ at GW $t$:

$$\text{tin}_{i,t} \geq x_{i,t} - x_{i,t-1}$$

$x_{i, t_0 - 1} = 1$ if player $i$ in prior squad, else $0$. Free-transfer conservation, cap 5:

$$\text{ft}_{t_0} = \text{ft}^{\text{init}}, \qquad \text{ft}_t = \min\!\bigl(5,\; 1 + \text{sv}_{t-1}\bigr) \text{ for } t > t_0$$

Transfer budget per GW — transfers in = free transfers used + hits; saved transfers ≤ held:

$$\sum_i \text{tin}_{i,t} = (\text{ft}_t - \text{sv}_t) + h_t, \qquad 0 \leq h_t \leq H_{\max}, \qquad \text{sv}_t \leq \text{ft}_t$$

Hit cost $C_h \cdot h_t$ in objective with $C_h = 6$ (raised from FPL's nominal 4 to discourage churn — replay runs at $C_h = 4$ took ~40 hits/season, top managers 4-8). Per-GW hit cap $H_{\max} = 1$ caps multi-hit gambles in a single GW.

**Banking incentive**: positive reward $\omega \cdot \text{sv}_t$ added to the objective per saved transfer (default $\omega = 0.3$, attenuated by RHC discount $\gamma^k$). Encodes option value of rolling — best-pick-next-week. Without this term, $\text{sv}_t = 0$ trivially because constraint slack carried no upside; solver burned every FT every GW. **Bank-leftover penalty** in cold-start `solve_initial_squad`: $+\beta \cdot \sum_i p_i x_i$ in the objective ($\beta = 0.5$) — flat EV cost per £1m unspent. Stops noisy early-season projections leaving £10m+ in bank when premium picks are only marginally above cheap ones in $\mu$.

### 5.5 Receding Horizon Control

Full multi-period MILP solved with $H = 8$ look-ahead each week. Per-GW objective contributions weighted by **geometric discount** $w_k = \gamma^k$ with $\gamma = 0.85$, indexed by horizon offset $k = t - t_0$ — canonical MPC form, monotone, smooth across horizon (no piecewise discontinuity). Profile: $[1.00, 0.85, 0.72, 0.61, 0.52, 0.44, 0.38, 0.32]$. Long-tail fixture-swing info feeds into transfer plan (Liverpool's brutal Apr–May, Arsenal's run-in) without trusting noisy point estimates 6+ GWs out as fully as next-GW prediction. Hit cost $-4 h_t$ **not** attenuated; -4 pt hit real regardless of horizon distance.

Only $t_0$ (next GW) decisions executed: `transfers_in`, `transfers_out`, `xi_ids`, `captain`, `vice`, `hits`. Next week re-solves from updated state (squad, bank, free transfers). Standard MPC formulation — see [Model Predictive Control: Recent Developments and Future Promise][ref-mpc] for stochastic + economic MPC survey.

---

## 6. Chip Scheduling

Lives in [src/chips.py](../src/chips.py). Chip activation = convex function of fixture quality the per-GW MILP doesn't see directly. Handled as greedy post-processing heuristic over same projection frame.

**FPL 2025/26 rules**: 8 chips total — two of each type. Set 1 (TC1, BB1, FH1, WC1) must be played by GW19 or it expires. Set 2 (TC2, BB2, FH2, WC2) becomes available GW20+. TC now multiplies captain by **3** (changed from ×2). FH cannot be played in consecutive GWs. WC + FH preserve banked transfers across activation.

| Chip | Heuristic |
|---|---|
| **Triple Captain** (×3 mult) | Pick GW + owned MID/FWD maximizing captaincy score $\kappa_{i,t}$ from §4.5: $t^{\star},\, i^{\star} = \arg\max_{t,\, i \in \text{squad},\, \text{pos}(i) \in \{3, 4\}}\, \kappa_{i,t}$ |
| **Bench Boost** | Pick GW with highest total bench EV: $t^{\star} = \arg\max_{t}\, \sum_{i \in \text{bench}} \mu_{i,t}$ |
| **Free Hit** | Pick GW with most teams blanking: $t^{\star} = \arg\max_{t}\, \lvert \{ k : k \text{ blanks at } t \} \rvert$. Constraint: not in consecutive GWs. |
| **Wildcard** | Trigger if RHC proposes ≥ 4 transfers IN or ≥ 2 hits — MILP's willingness to pay hits = proxy signal current squad far from optimal. |

**Replay simulation** ([src/season_replay.py](../src/season_replay.py)): TC + BB are activated greedily in the per-half budget — TC fires when $q_{90,\text{cap}} - \mu_{\text{cap}} \geq 4.5$, BB when $\sum_{i \in \text{bench}} \mu_i \geq 10.0$, with force-activation at GW19 (set 1 deadline) and GW38 (set 2 deadline) to avoid expiry. WC + FH defer (require re-solving the squad inside the replay loop).

---

## 7. Validation and Calibration

Lives in [src/backtest.py](../src/backtest.py) + [src/calibration.py](../src/calibration.py). Covers all three model heads (points, match, minutes). Run: `python src/backtest.py --k 5` or `--start S --end E`. Output → `data/processed/backtest/`: per-arm prediction CSVs, per-quantile coverage/pinball/Brier, single human-readable `report.md`.

### 7.1 Walk-forward CV

Per holdout GW $G$: retrain on rows with `round` $< G$, predict `round` $= G$, accumulate predictions. Rolling features in §2 shift-1 partitioned, so feature frame built once on full history is leakage-free as long as target-bearing rows at `round` $\geq G$ excluded from training. Splits by `round` after feature construction rather than rebuilding features per holdout.

### 7.2 Points calibration

Two scopes reported per (position, quantile):

- **`all`** — every (player, GW) row including DNPs (y = 0 inflates lower-tail coverage).
- **`played`** — `minutes > 0`. Production-conditional view that matters for ranking starters, since engine multiplies raw quantiles by expected availability before optimizer.

Per quantile: empirical coverage $P(y \leq \hat{q}_\alpha)$, coverage gap from nominal $\alpha$, pinball loss. Aggregated across positions in position-pooled overall row. Played-only view = input to `recalibrate.py` in §4.6.

### 7.3 Match calibration

Per side (home/away) on held-out fixtures: mean per-marginal Poisson NLL, goal MAE, Brier on clean-sheet probabilities, predicted vs actual CS rate. `cs_rate_gap` reports calibration of marginal Poisson goal models — **not** a lever for tuning DC ρ, since τ correction conserves row/column marginals exactly (§3.2). Tune ρ on joint-score NLL via [src/tune_dc_rho.py](../src/tune_dc_rho.py) instead.

### 7.4 Minutes-model audit

Walk-forward arm scores held-out minutes/90 + binary `played`: MAE, ROC-AUC, Brier, 10-bin reliability table (predicted-mean vs actual-played-rate per bucket). Most recent run (GW 32–35) shows AUC ≈ 0.93 — strong rank order — but predicted played rate 0.24 vs actual 0.34, negative reliability gap in every bucket, worst miss in 0.1–0.5 mid-probability range. Engine multiplies raw quantiles by `mins_pred`, so under-prediction systematically depresses rotation-risk premium projections. Per-position isotonic recalibration in §4.7 closes gap; pass `--minutes-recalib data/minutes_recalib.json` to `backtest.py` to re-run audit with calibrated predictions.

---

## 8. Repository Layout

```
fpl-ml-manager/
├── src/
│   ├── main.py                  # Orchestrator + markdown report writer
│   ├── data_loader.py           # FPL-Core-Insights CSV loader + live FPL API price overlay
│   ├── features.py              # ClubElo + rolling team/player + per-player Opta features
│   ├── train_match_model.py     # Poisson goals + DC τ + analytic CS
│   ├── train_points_model.py    # Per-position Quantile XGBoost (12 boosters)
│   ├── train_minutes_model.py   # Two-stage: P(plays) classifier × E[mins/90 | plays] regressor
│   ├── fpl_engine.py            # Inference engine, projection frame builder
│   ├── optimizer.py             # MILP squad + XI + captain, RHC transfers
│   ├── chips.py                 # TC / BB / FH / WC heuristics
│   ├── backtest.py              # Walk-forward CV harness for points / match / minutes
│   ├── calibration.py           # Coverage / pinball / Brier / reliability tables
│   ├── recalibrate_points.py    # Per-(pos, quantile) affine recalibration for points head
│   ├── recalibrate_minutes.py   # Per-pos isotonic recalibration for minutes / 90 head
│   ├── league_table.py          # Pre-match standings + tier-distance stakes features
│   ├── season_replay.py         # GW-by-GW backtest harness with chip simulation
│   └── tune_dc_rho.py           # Grid-search Dixon-Coles ρ on walk-forward CS Brier
├── data/
│   ├── players.csv, teams.csv, fixtures.csv, history.csv
│   ├── .fpl_ci_cache/                    # raw FPL-CI per-GW snapshots (cache only)
│   ├── xgb_home_goals.json, xgb_away_goals.json
│   ├── xgb_points_q{10,50,90}_p{1,2,3,4}.json   # 4 positions × 3 quantiles
│   ├── xgb_minutes.json                  # minutes / 90 model
│   ├── points_recalib.json               # affine recalib coefficients (auto-loaded)
│   ├── minutes_recalib.json              # per-pos isotonic knots (auto-loaded)
│   └── processed/
│       ├── lineup.md            # Weekly markdown report (generated)
│       ├── squad_snapshot.csv   # Carried state for next RHC pass
│       └── backtest/            # Walk-forward predictions + calibration tables + dc_rho_grid.csv
├── docs/
│   ├── README.md                # This file
│   └── FPL_101.md               # Domain primer
└── .github/workflows/
    ├── pipeline.yml             # GitHub Actions: 05:30 + 17:30 UTC daily
    └── season_replay.yml        # workflow_dispatch GW-by-GW backtest
```

---

## 9. Installation and Usage

### Requirements

- Python 3.11+
- No GPU required — XGBoost CPU fast enough for dataset size

### Local setup

```bash
# 1. Clone
git clone https://github.com/truong-tt/fpl-ml-manager
cd fpl-ml-manager

# 2. Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline
python src/main.py
```

First run trains every model artifact from scratch. Subsequent runs reuse `data/*.json` + retrain only when file missing. Output → [data/processed/lineup.md](../data/processed/lineup.md). Carried squad state at [data/processed/squad_snapshot.csv](../data/processed/squad_snapshot.csv), consumed by next RHC pass.

### Tuning and recalibration

Recalibration is automatic. `python src/main.py` calls `_maybe_recalibrate(...)` which walk-forward-retrains the points + minutes heads when `data/points_recalib.json` / `data/minutes_recalib.json` are older than `RECALIB_STALE_DAYS` (= 14) or missing. JSONs auto-load via `predict_quantiles` / `predict_minutes` on next inference.

Manual triggers (debugging, custom holdouts, DC ρ tuning):

```bash
# Force-refresh: delete the JSON and re-run main.py
rm data/points_recalib.json data/minutes_recalib.json
python src/main.py

# Standalone walk-forward CV → backtest predictions + calibration tables
python src/backtest.py --k 8

# Standalone refit (consumes data/processed/backtest/*.csv from previous step)
python src/recalibrate_points.py
python src/recalibrate_minutes.py

# Grid-search Dixon-Coles ρ (§3.2). NOT automated — manually update DC_RHO in train_match_model.py
python src/tune_dc_rho.py --k 8

# Re-run backtest with recalibrated minutes head to verify gap closure
python src/backtest.py --k 8 --minutes-recalib data/minutes_recalib.json
```

### Scheduled runs

GitHub Actions workflow [.github/workflows/pipeline.yml](../.github/workflows/pipeline.yml) runs **twice daily** at 05:30 + 17:30 UTC — 30 min after FPL-Core-Insights upstream refresh (~05:00 + 17:00 UTC). Pipeline is idempotent: only commits when artifacts changed. `concurrency: fpl-update` group prevents overlap if a run spans the next slot.

Recalibration auto-fires inside `python src/main.py` when `data/points_recalib.json` or `data/minutes_recalib.json` exceeds `RECALIB_STALE_DAYS` (= 14) — walk-forward retrain on last `RECALIB_HOLDOUT_K` (= 8) finished GWs. Manual override: delete the JSON to force re-fit on next run.

---

## 10. Recently Shipped

- **Season replay backtest harness + chip simulation** ([src/season_replay.py](../src/season_replay.py), [.github/workflows/season_replay.yml](../.github/workflows/season_replay.yml)). Walk-forward by GW from a configurable start: filter history to `round < G` in the current season, build engine, cold-start at first GW (`solve_initial_squad`) then RHC after, score against actual `total_points`. GW1 graft: most recent per-player row from prior seasons relabelled `season=SEASON` so `_latest_rolling` has a baseline (matches manager intuition for new-season cold starts — last-season xG/xA, transfer noise accepted). FPL 2025/26 chip rules wired in: 2× each chip, set 1 expires at GW19, set 2 valid GW20-38, TC = ×3, FH non-consecutive. TC + BB activate greedily per-half (force-fire at deadlines to avoid expiry); WC + FH deferred (need re-solving inside the loop). `workflow_dispatch` triggers a full Actions run; results pushed back to repo as `data/processed/season_replay.{md,csv}`.
- **Hit cap, banking incentive, bank-leftover penalty** ([src/optimizer.py](../src/optimizer.py)). `solve_rhc_transfers` now defaults to `hit_cost = 6` (raised from FPL's nominal 4 — earlier replays burned ~40 hits/season for -160 net) with `hit_cap_per_gw = 1` hard ceiling. New `bank_transfer_weight = 0.3` term `+ω · sv_t` rewards rolling FTs (constraint slack alone gave the solver no upside in saving). `solve_initial_squad` adds `bank_penalty = 0.5` flat EV cost per £1m unspent — stops noisy early-season cold-start parking £10m+ in bank when premium picks barely outscore cheap ones.
- **Stakes features: tier-distance + late-season motivation** ([src/league_table.py](../src/league_table.py)). Pre-match standings reconstructed per (season, event) from finished fixtures with `cumsum().shift(1)` on a (season × team × 1..38 GW) grid so opponents' state is available even on blank GWs. New cols (per side, signed-gap normalised by `3 × gws_remaining`): `pts_to_title_norm` (1st), `pts_to_top4_norm` (UCL), `pts_to_top6_norm` (UEL/UECL band), `pts_to_safety_norm` (17th), `in_drop_zone`, `rank`, `ppg`, `gws_remaining`. Match model gets `h_*` + `a_*`; points model side-conditions to `own_*` / `opp_*`. Encodes "Arsenal title-chasing fully engaged" / "WHU mid-table on beach mode" — structural signal EMA rolling form lags on by halflife=2.5 GW. Schema drift guard auto-retrains heads on next pipeline run.
- **Feature-schema drift retrain guard** ([src/main.py](../src/main.py)). `_ensure_models` now probes each cached booster's stored `feature_names` via `_booster_features` and forces a retrain when it diverges from the current `features.py` output (`_schema_drift`). Retrain also wipes the matching recalib JSON so `_maybe_recalibrate` re-fits against fresh raw quantiles. Fixes XGBoost `feature_names mismatch` on GitHub Actions runs that always pull HEAD's stale committed artifacts.
- **GW-in-play pipeline guard** ([src/main.py](../src/main.py)). `_gw_in_play` short-circuits `main()` when any current-season fixture is live (kickoff in [now − 3 h, now], `finished=False`) or imminent (kickoff in [now, now + 2 h]). Prevents producing a lineup the user can't act on between deadline and final whistle. GitHub Actions cron (`30 5,17 * * *`) still fires unconditionally; guard exits early after checkout + setup.
- **EMA rolling features** (§2.2, [src/features.py](../src/features.py)). Replaced `rolling(w).mean()` with `ewm(halflife=w/2).mean()` across all team/Opta/player rolling stats. Half-life scales with nominal window (`roll5` → halflife 2.5, `roll10` → halflife 5). Catches manager/set-piece/form regime changes faster — one-pre-GW shock now propagates exponentially rather than waiting `w` GWs to slide out of equal-weight window. Forces one-time retrain of every model artifact.
- **Two-stage minutes head** (§4.3, [src/train_minutes_model.py](../src/train_minutes_model.py)). Replaces single `reg:logistic` regressor with `binary:logistic` `plays` classifier × `reg:logistic` `mins-given-played` regressor on disjoint subsets of bimodal minutes target. Engine consumes `mins_pred = plays × mins_when_played` for q10/q50 (mean-mass needs minutes on pitch), `plays` alone for q90 (hauls usually land before any sub, ceiling near-fully realised given player gets on). Legacy single-head artifact stays loadable as fallback.
- **Isotonic per-(position, quantile) points recalibration** (§4.6, [src/recalibrate_points.py](../src/recalibrate_points.py)). Equal-frequency bins on `q_pred` × per-bin empirical α-quantile of `y` × `sklearn.isotonic.IsotonicRegression` — canonical non-parametric quantile recalibrator. Falls back to prior affine map for any (pos, α) cell with fewer than `MIN_ROWS_ISOTONIC` (= 400) rows. Non-crossing re-enforced row-wise.
- **Bonus-head additive blend** (§4, [src/train_bonus_model.py](../src/train_bonus_model.py)). Three quantile boosters (`reg:quantileerror`, α ∈ {0.10, 0.50, 0.90}) trained directly on FPL `bonus` column. Engine adds `BONUS_BLEND = 0.5 ×` predicted bonus quantiles onto points-head quantiles before availability multiplier — lifts ceiling specifically for bonus-heavy archetypes (CS-keeping defenders, save-rich GKs) without disturbing q10 floor. Half-blend pending clean retrain of points head on `total_points - bonus`.
- **Joint-score Monte-Carlo aggregation** (§4, [src/fpl_engine.py](../src/fpl_engine.py)). Per-fixture `(player, team)` rows summed under correlated draws: `MC_TEAM_RHO = 0.4` of per-row dispersion shared with single team-shock per (team, GW) per draw, rest idiosyncratic. Captures Salah+Virgil-rise-together / Salah+Diaz-rise-together correlation prior independent aggregation ignored. Outputs same `xp_t / var_t / cap_xp_t` schema. `mc_samples = 0` falls back to deterministic Swanson aggregation.
- **Twice-daily schedule + auto-recalibration** ([src/main.py](../src/main.py), [.github/workflows/pipeline.yml](../.github/workflows/pipeline.yml)). Workflow cron `30 5,17 * * *` UTC — fires 30 min after each FPL-Core-Insights upstream refresh window. `_maybe_recalibrate` re-fits `points_recalib.json` + `minutes_recalib.json` when stale (> `RECALIB_STALE_DAYS` = 14 d) by running walk-forward CV in-process and calling `fit_points_recalib` + `fit_minutes_recalib` directly — no manual `python src/backtest.py` step required. `concurrency: fpl-update` prevents overlap.
- **Geometric RHC discount** (§5.5, [src/optimizer.py](../src/optimizer.py)). Replaced piecewise `[1,1,1,1,1,0.6,0.4,0.2]` with $\gamma^k$, $\gamma = 0.85$ over $H = 8$. Smooth, monotone, MPC-canonical. No discontinuity at $k = 5$. Constants `RHC_DISCOUNT` + `RHC_HORIZON` exposed for tuning.
- **Swanson (Keefer–Bodily) mean estimator for $\mu_{i,t}$** (§4.4). Replaced Simpson weights $(1, 4, 1)/6$ — earlier mislabeled "Pearson–Tukey" — with skew-aware $0.3\,\hat{q}_{10} + 0.4\,\hat{q}_{50} + 0.3\,\hat{q}_{90}$. Heavy median weight under-fired tail mean on right-skewed FPL distribution; Swanson 0.3/0.4/0.3 calibrated to lognormal-family, lifts $\mu$ ~12 % on premium archetypes without disturbing rank order. Real Pearson–Tukey (1965) needs q05/q95 boosters that pinball-loss noise prohibits at 3k rows/position. Risk term remains linear in $\hat{s}$ for CBC LP compatibility.
- **Minutes-head isotonic recalibration** (§4.7, [src/recalibrate_minutes.py](../src/recalibrate_minutes.py)). Closes §7.4 reliability gap. Re-fit drops minutes Brier 0.122 → 0.095, aligns predicted played-rate (0.39 vs actual 0.39) on GW 27–35 holdout.
- **Dixon-Coles ρ tuning fixed** (§3.2, [src/tune_dc_rho.py](../src/tune_dc_rho.py)). Two bugs found: (a) `_dc_tau` captured `DC_RHO` as default arg evaluated at def time, so module-level monkeypatch did nothing; (b) prior tuning metric `cs_brier` mathematically invariant to ρ because τ conserves row/column marginals. Now: late-binding ρ lookup + grid-search by joint-score NLL on actual outcomes. ρ now empirically identifiable; default ρ=-0.10 retained pending larger holdout (GW 27–35 favours ρ≈0 by 0.007 nats — within sampling noise on 136 fixtures).

---

## 11. Future Work

Open items, ordered by impact:

1. **Retrain points head on `total_points − bonus`.** Removes double-count between bonus head + points head's `roll_bps` features so `BONUS_BLEND` can move 0.5 → 1.0. Cleanest decoupling of ceiling-driving logic from main scoring head.
2. **Rank-EV objective via end-of-season simulation.** Extend joint Monte-Carlo aggregator to simulate full remaining-season trajectories per squad/transfer plan. Replace MILP's points-EV objective with EO-weighted percentile, calibrated against population EO. Points-EV → rank-EV conversion non-linear; current `lambda_eo` differential tilt approximates only first-order.
3. **Correlated-risk portfolio objective (MIQP).** Replace diagonal linear $-\nu \hat{s}_{i,t}$ penalty in §5.2 with full quadratic $x^\top \Sigma x$ — Σ now estimable from MC draws in §4 — once MIQP solver (Gurobi/CPLEX/SCIP) wired in. CBC LP-only.
4. **Set-piece and manager regime changes.** Embedding-based detection of regime breaks (new manager, new set-piece taker) invalidating historical rolling features. EMA features partially help by weighting recent observations more, but step-change still bleeds through half-life window.
5. **Learned chip scheduler.** Re-formulate chip activation as jointly-solved MILP extension rather than post-hoc heuristics in §6. Chip-EV path-dependent on transfer plan + DGW timing.
6. **WC + FH simulation in season replay.** §6 describes both, but `season_replay.py` only simulates TC + BB. WC requires re-solving the squad with unlimited free transfers; FH solves a one-GW temp squad and reverts. Both straightforward extensions of `solve_initial_squad` / `solve_rhc_transfers` — deferred only because they need re-solving inside the replay loop.
7. **Strict walk-forward retraining in season replay.** Today the replay reuses production heads (trained on full season). Per-GW retrain on `round < G` would remove parameter-level leakage and produce an honest out-of-sample number. Cost ~30 sec/GW × 36 GW = ~18 min/run.
8. **Empirical MC team-ρ.** `MC_TEAM_RHO = 0.4` in §4.9 hand-set. Fit per-position-pair correlation from history: residualise actual points by predicted $\mu_{i,t}$, compute correlation of residuals across same-team-same-GW player pairs. Likely position-pair dependent (CB-CB high under CS, GK-FWD low). Constant ρ proxy until then.

---

## 12. References

**Boosting and quantile regression**

- [XGBoost: A Scalable Tree Boosting System][ref-xgboost] — Chen & Guestrin, *KDD* 2016. Backbone for Poisson goal model + three quantile point models.
- [Regression Quantiles][ref-koenker] — Koenker & Bassett, *Econometrica* 1978. Origin of pinball/check loss that XGBoost's `reg:quantileerror` objective minimizes for q10/q50/q90 boosters.
- [Quantile and Probability Curves Without Crossing][ref-chernozhukov] — Chernozhukov, Fernández-Val & Galichon, *Econometrica* 2010. Principled non-crossing alternative to row-sort heuristic used here.

**Football scoring models**

- [Modelling Association Football Scores and Inefficiencies in the Football Betting Market][ref-dc] — Dixon & Coles, *Journal of the Royal Statistical Society* 1997. Source of low-score $\tau$ correction applied to joint Poisson PMF.
- [Bayesian hierarchical model for the prediction of football results][ref-baio] — Baio & Blangiardo, *Journal of Applied Statistics* 2010. Modern Bayesian hierarchical Poisson goal-scoring model — alternative considered + rejected for single-season sample-size reasons.

**Ratings**

- [Using ELO ratings for match result prediction in association football][ref-hvattum] — Hvattum & Arntzen, *International Journal of Forecasting* 2010. Adapts Elo chess → football, validates against bookmaker odds.
- [How We Calculate NBA Elo Ratings][ref-538] — Silver & Fischer-Baum, FiveThirtyEight 2015. Source of margin-of-victory multiplier idea.

**Optimization**

- [Picking Winners in Daily Fantasy Sports Using Integer Programming][ref-hvz] — Hunter, Vielma & Zaman, 2016. Direct precedent for applying portfolio-style integer programming to fantasy sports lineup selection; motivates $\mu - \nu \sigma^2$ structure of squad objective in FPL setting.
- [Model Predictive Control: Recent Developments and Future Promise][ref-mpc] — Mayne, *Automatica* 2014. Modern survey of MPC incl. stochastic/economic variants; canonical reference for Receding Horizon Control structure used here.
- [PuLP: A Linear Programming Toolkit for Python][ref-pulp] — Mitchell, O'Sullivan & Dunning, 2011. Modeling layer over COIN-OR CBC solver used here.

**Data sources**

- [olbauday/FPL-Core-Insights][ref-fpl-ci] — primary dataset. CSVs combining FPL API + Opta-like per-match stats + ClubElo ratings, refreshed 2×/day. Used files: `teams.csv`, `players.csv`, `By Gameweek/GW{n}/{fixtures, playerstats, player_gameweek_stats, playermatchstats}.csv`, `gameweek_summaries.csv`.
- [Fantasy Premier League Public API][ref-fpl] — live overlay. Used only for `bootstrap-static/` endpoint to refresh current-GW prices, ownership, status, `chance_of_playing_next_round` (FPL-CI lags by up to 12h).
- [ClubElo][ref-clubelo] — Elo ratings exposed on FPL-CI's fixtures.csv (`home_team_elo`, `away_team_elo`).

[ref-xgboost]: https://arxiv.org/abs/1603.02754
[ref-dc]: https://www.ajbuckeconbikesail.net/wkpapers/Airports/MVPoisson/soccer_betting.pdf
[ref-baio]: https://doi.org/10.1080/02664760802684177
[ref-koenker]: https://people.eecs.berkeley.edu/~jordan/sail/readings/koenker-bassett.pdf
[ref-chernozhukov]: http://alfredgalichon.com/wp-content/uploads/2012/10/Econometrica_article_may-2010.pdf
[ref-538]: https://fivethirtyeight.com/features/how-we-calculate-nba-elo-ratings/
[ref-hvz]: https://arxiv.org/abs/1604.01455
[ref-mpc]: https://doi.org/10.1016/j.automatica.2014.10.128
[ref-hvattum]: https://www.sciencedirect.com/science/article/abs/pii/S0169207009001708
[ref-pulp]: https://github.com/coin-or/Cbc
[ref-fpl]: https://fantasy.premierleague.com/api/
[ref-fpl-ci]: https://github.com/olbauday/FPL-Core-Insights
[ref-clubelo]: https://clubelo.com

---

## Data and Credit

All credit upstream → providers in §12. Use subject to each provider's own terms. Project consumes only public, read-only endpoints/files. Not affiliated with or endorsed by Premier League, ClubElo, or FPL-Core-Insights maintainers.

---

## License

TBD.

> **Note:** FPL API, FPL-Core-Insights dataset, ClubElo data governed by their own separate terms.
