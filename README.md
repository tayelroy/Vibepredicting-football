
# vibepredicting-worldcup26

An advanced, player-isolated football predictive engine built to simulate and forecast the 2026 World Cup matchups. Moving beyond primitive historical international team data, this model synthesizes live "Synthetic National Teams" by tracking and aggregating the real-time domestic performance metrics of individual players.

Powered by a two-stream data architecture, live commercial feeds from the **Sportradar Soccer Extended v4 API**, and a modified **Dixon-Coles Poisson optimization framework**, this engine delivers highly precise scoreline probability matrices.

---

## Technical Architecture Overview

Predicting international tournaments using historical national team data is inherently flawed due to massive time lags, tactical shifts, and sparse match samples. `vibepredicting-worldcup26` solves this by building national profiles from the ground up utilizing a **Targeted 1/11 Player Match Log Aggregation Pipeline**.

---

## ⚙️ Core Data Features & Proxies

Due to commercial trial API constraints, raw spatial event loops (passes, carries, pressures) are selectively augmented by a robust **Macro-Proxy Ingestion Layer** extracted natively from Sportradar's player-match statistics:

### Stream A: Additive Player Metrics
Individual player outputs are isolated, calculated on a per-match basis, normalized to a per-90 rate (`(stat / minutes_played) * 90`), and scaled via a strict **League Strength Coefficient** (e.g., Premier League = 1.0, Saudi Pro League = 0.60):
*   **Synthetic npxG:** Calculated using an advanced shot-accuracy distribution model:  
    $$\text{npxG Proxy} = (\text{Shots on Target} \times 0.30) + ((\text{Shots Total} - \text{Shots on Target}) \times 0.05)$$
*   **xT Proxy (Ball Progression):** Captures creative penetration using `chances_created + dribbles_completed`.
*   **Deep Progressions (Box Entries):** Tracks final-third threat using `chances_created + corner_kicks`.

### Stream B: Systemic On-Pitch Metrics
Defensive metrics cannot be accumulated purely additively without causing mathematical statistical inflation. Pressing and defensive attributes are evaluated using the team's metrics *while the player was on the pitch*, mapped against a strict `POSITIONAL_WEIGHTS` constraint matrix:
*   **PPDA Proxy (Pressing Intensity):** Formulated as:  
    $$\text{PPDA} = \frac{100 - \text{average\_ball\_possession}}{\text{tackles\_total} + \text{interceptions} + \text{fouls\_committed}}$$
*   **Shot Suppression:** Parsed from goalkeeper `shots_faced` logs to proxy open-play shots allowed per 90.

---

## 🗂️ Project Structure

*   `roster.json`: Stores the target 22-man starting lineups mapped to their active domestic club and role configurations.
*   `match_log_processor.py`: Connects to the Sportradar v4 endpoint, reads the nested `statistics` objects, normalizes values by `minutes_played`, and manages the targeted "Sniper" match sampling (fetching the last 8–10 games per player to capture current form).
*   `module_synthetic_aggregation.py`: The main pandas aggregation engine. Diverges into Stream A and Stream B calculation flows, applying league filters and returning the final tactical row for the national team.
*   `dixon_coles_engine.py`: Ingests the synthetic team profiles, fits the low-scoring draw adjustment parameter ($\rho$), and computes independent Poisson distributions ($\lambda$ and $\mu$) via `scipy.optimize`.

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.10+ installed along with the required analytical stack:
```bash
pip install pandas numpy requests scipy

2. Configure Environment Keys
Export your Sportradar commercial API developer trial key:
export SPORTRADAR_API_KEY="your_v4_trial_key_here"

3. Run the Pipeline
Orchestrate the Targeted Sniper scrape and output the World Cup scoreline probability matrix:
python dixon_coles_engine.py

🔬 The Math Behind the Prediction: Dixon-Coles ELI5
Standard Poisson models assume the number of goals Team A scores is entirely independent of Team B. Real football doesn't work that way. This project utilizes the Dixon-Coles Model (1997) to adjust for two critical variables:
1. The Draw Bias (\rho): In low-scoring games past the 80th minute, teams become risk-averse, leading to a higher rate of draws than standard math predicts. The model applies an interdependence factor (\rho) to artificially scale up the likelihood of tight scorelines like 0-0, 1-1, 1-0, and 0-1.
2. League Relativity: A goal contribution generated in the Saudi Pro League does not share parity with a goal contribution generated against a Premier League low block. The aggregation engine normalizes player quality across distinct domestic competitive contexts using UEFA-coefficient-scaled multipliers.