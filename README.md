# Vibepredicting Football

Bivariate Poisson predictive engine for simulating international football fixtures using real-time player club form and a two-stream starting lineup aggregator.

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen?style=flat-square)](https://github.com/tayelroy/Vibepredicting-football)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)

---

## Overview

Predicting international tournaments using historical national team match results is statistically flawed. National teams play sparse fixtures, suffer from significant squad turnover, and experience massive time lags between competitive windows. This project resolves these data gaps by building synthetic national team profiles from the ground up, utilizing a targeted 1/11 starting lineup player match log aggregation pipeline.

By pulling rolling domestic performance data directly from the Sportradar Soccer Extended v4 API, the engine isolates the current club form of each player in a starting lineup. These metrics are processed through a two-stream aggregation framework (separating individual physical output from team-level systemic solidity) and normalized for league difficulty. The resulting covariates dynamically adjust a time-decayed Dixon-Coles bivariate Poisson model, producing high-fidelity scoreline probability distributions based on the actual players taking the pitch today.

### Technical Architecture

The following diagram illustrates the ingestion, aggregation, and prediction pipeline:

![Technical Architecture](https://github.com/user-attachments/assets/5ccd5145-c06a-4153-b4d8-ca9c69adaf3b)

---

## Key Features

* **Targeted 1/11 Player-Log Ingestion**: Isolates individual form over a rolling 8-match domestic window using a targeted request strategy to optimize API quota usage.
* **Decoupled Two-Stream Aggregation**: Sums pure additive metrics (non-penalty expected goals, threat creation) while distributing systemic team metrics (pressing intensity, defensive shot suppression) based on canonical positional weights.
* **League Relativity Normalization**: Standardizes performance metrics across leagues using difficulty multipliers (e.g., Premier League = 1.0, Saudi Pro League = 0.60).
* **Vectorized Dixon-Coles Solver**: Fits parameters via a high-performance vectorized log-likelihood function using SciPy optimization, reducing model training times to under 0.1 seconds.
* **Dynamic Covariate Scaling**: Adjusts baseline attack and defense strengths log-linearly based on active player form before simulating scorelines.
* **Automated Master Orchestrator**: Detects matchups from the lineup configuration, checks the training database validity, runs pipeline steps, and outputs serialized predictions.

---

## Prerequisites & System Requirements

To run this prediction engine, you must meet the following requirements:

* **Python**: Version 3.10 or higher.
* **Sportradar API Key**: Access to the Sportradar Soccer Extended v4 API (I used the Developer Trial).

### Required Python Libraries
* `pandas` (Data manipulation)
* `numpy` (Numerical operations)
* `scipy` (Parameter optimization and probability distributions)
* `requests` (API requests)
* `python-dotenv` (Local environment management)

---

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/tayelroy/Vibepredicting-football.git
   cd Vibepredicting-football
   ```

2. **Install Dependencies**
   ```bash
   pip install pandas numpy scipy requests python-dotenv
   ```

3. **Configure Environment Keys**
   Create a `.env` file in the root directory of the project and add your Sportradar API key:
   ```env
   sportradar_api_key="your_actual_sportradar_v4_key_here"
   ```

---

## Usage

### 1. Configure the Match Lineups
Open `roster.json` in the root folder and define the starting lineups for the two teams you want to simulate. The dictionary keys must end in `_xi` (e.g., `spain_xi`, `portugal_xi`), mapping each player to their domestic club, role, and league:

```json
{
  "spain_xi": [
    {
      "player_name": "Lamine Yamal",
      "club": "Barcelona",
      "league": "La Liga",
      "role": "Right Wing"
    }
  ],
  "portugal_xi": [
    {
      "player_name": "Cristiano Ronaldo",
      "club": "Al Nassr",
      "league": "Saudi Pro League",
      "role": "Striker"
    }
  ]
}
```

### 2. Execute the Simulation
Run the master orchestrator script to map player IDs, fetch match statistics, compile the training data, and run the Dixon-Coles model:
```bash
python3 run_prediction.py
```

### 3. Expected Terminal Output
When execution completes, you will see a detailed matching report printed to your console:

```text
==================================================
      WC 2026 PREDICTION ENGINE RUNNER
==================================================
Active Matchup : Spain vs. Portugal

 Running: pipeline/discover_ids.py
Saved ID mappings to output/roster_ids.json

 Running: pipeline/process_match_logs.py
Saved player profiles to output/player_profiles.json
Saved player metrics to output/player_metrics.csv

 Running: aggregation/aggregate_national_teams.py
Saved synthetic team profiles to output/synthetic_national_teams.csv
Saved detailed synthetic contributions to output/synthetic_aggregation_details.json

 Running: predict.py

--- Fitting Dixon-Coles Model ---
Fitting model for 119 unique national teams across 381 matches...
Running optimization (SLSQP)...
Optimization complete.
  Fitted Home Advantage (gamma): 1.3357
  Fitted Low-Score Correlation (rho): -0.3000

--- Covariates (Synthesized Form) ---
Spain:    Attack Covariate = 2.3627 | Defense Covariate = 9.3195
Portugal:    Attack Covariate = 2.3116 | Defense Covariate = 10.2795

==================================================
      MATCH PREDICTION REPORT
==================================================

1. BASELINE MODEL (Historical Results Only):
  Expected Goals: Spain 1.48 - 1.01 Portugal
  Spain Win:      44.18%
  Draw:            33.62%
  Portugal Win:   22.20%

2. ADJUSTED MODEL (Incorporating 1/11 Starting Lineup Form):
  Expected Goals: Spain 3.18 - 2.08 Portugal
  Spain Win:      58.90%
  Draw:            18.21%
  Portugal Win:   22.90%

Top 5 Most Likely Scorelines:
  1. Spain 3 - 2 Portugal  |  Probability: 6.02%
  2. Spain 3 - 1 Portugal  |  Probability: 5.79%
  3. Spain 2 - 2 Portugal  |  Probability: 5.69%
  4. Spain 2 - 1 Portugal  |  Probability: 5.47%
  5. Spain 4 - 2 Portugal  |  Probability: 4.78%

Saved prediction results to output/match_prediction.json
==================================================
```

---

## Configuration & API Reference

### Data Directory Structures
* **`pipeline/discover_ids.py`**: Queries Sportradar Competitors to map player names and club rosters to their specific API resource IDs.
* **`pipeline/process_match_logs.py`**: Fetches the rolling 8 summaries per player, extracts statistics, resolves minutes played using a fallback estimator for sub-appearances, and compiles averages.
* **`pipeline/build_intl_results.py`**: Fetches the historical match summaries for target nations to build the Dixon-Coles training corpus.
* **`aggregation/aggregate_national_teams.py`**: Accumulates player profiles into national team metrics using additive (Stream A) and weighted (Stream B) channels.

### Model Parametrization
The prediction script uses the following hyperparameter settings:
* **Time Decay Factor ($\xi$)**: `0.2`. Controls how heavily past matches are discounted over time (e.g., $w = e^{-\xi \times t}$).
* **Attack Scaling ($\theta_a$)**: `0.15`. Log-linear adjustment scalar for the attacking form covariate.
* **Defense Scaling ($\theta_d$)**: `0.04`. Log-linear adjustment scalar for the defensive shot-suppression covariate.

---

## Contributing Guidelines

Contributions are welcome. Please adhere to the following workflow:

1. **Feature Branching**: Do not commit directly to the `main` branch. Create short-lived descriptive feature branches (e.g., `feat/model-optimization`, `fix/import-paths`).
2. **Coding Standards**: Maintain PEP 8 styling. Document helper functions and ensure all calculations are vectorized where possible.
3. **Data Safety**: Never stage or commit folders containing raw API JSON responses (`cache/`) or compiled statistics (`output/`). Ensure your local `.gitignore` is active before committing.
4. **Pull Requests**: Open a pull request against `main` and provide execution logs verifying the changes compile without side effects.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
