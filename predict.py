import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from models.dixon_coles import DixonColesModel

def load_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text())

def print_exact_scores(matrix, home_team, away_team, top_n=5):
    """
    Finds and prints the top N most likely exact scorelines.
    """
    scores = []
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            scores.append((x, y, matrix[x, y]))
            
    # Sort by probability descending
    scores = sorted(scores, key=lambda x: x[2], reverse=True)
    
    print(f"\nTop {top_n} Most Likely Scorelines:")
    for idx, (x, y, p) in enumerate(scores[:top_n]):
        print(f"  {idx+1}. {home_team} {x} - {y} {away_team}  |  Probability: {p*100:.2f}%")

def main():
    # Load files
    intl_path = Path(__file__).parent / "output" / "intl_results.csv"
    synthetic_path = Path(__file__).parent / "output" / "synthetic_national_teams.csv"
    
    if not intl_path.exists():
        print("Error: output/intl_results.csv is missing. Run build_intl_results.py first.")
        return
    if not synthetic_path.exists():
        print("Error: output/synthetic_national_teams.csv is missing. Run aggregate_national_teams.py first.")
        return
        
    df_intl = pd.read_csv(intl_path)
    df_teams = pd.read_csv(synthetic_path)
    
    # Fit Dixon-Coles
    model = DixonColesModel()
    model.fit(df_intl, xi=0.2)
    
    # Extract covariates for Spain and Portugal
    # We define:
    #   attack_cov = npxG + xT_Proxy * 0.1
    #   defense_cov = Shot_Suppression (lower is better, higher means more shots conceded/concession rate)
    teams = df_teams["team"].tolist()
    if len(teams) < 2:
        print("Error: output/synthetic_national_teams.csv must contain at least 2 teams.")
        return
    team1_name, team2_name = teams[0], teams[1]
    
    team1_row = df_teams[df_teams["team"] == team1_name].iloc[0]
    team2_row = df_teams[df_teams["team"] == team2_name].iloc[0]
    
    team1_covs = {
        "attack": team1_row["npxG"] + team1_row["xT_Proxy"] * 0.1,
        "defense": team1_row["Shot_Suppression"]
    }
    team2_covs = {
        "attack": team2_row["npxG"] + team2_row["xT_Proxy"] * 0.1,
        "defense": team2_row["Shot_Suppression"]
    }
    
    # Calculate means to mean-center the covariates (avoid double-counting baseline strength)
    df_teams["attack_raw"] = df_teams["npxG"] + df_teams["xT_Proxy"] * 0.1
    mean_attack = df_teams["attack_raw"].mean()
    mean_defense = df_teams["Shot_Suppression"].mean()
    
    team1_covs["attack"] -= mean_attack
    team1_covs["defense"] -= mean_defense
    team2_covs["attack"] -= mean_attack
    team2_covs["defense"] -= mean_defense
    
    print("\n--- Covariates (Synthesized Form - Mean Centered) ---")
    print(f"{team1_name}:    Attack Covariate = {team1_covs['attack']:.4f} | Defense Covariate = {team1_covs['defense']:.4f}")
    print(f"{team2_name}:    Attack Covariate = {team2_covs['attack']:.4f} | Defense Covariate = {team2_covs['defense']:.4f}")
    
    # Predict baseline (no covariates)
    baseline_pred = model.predict_probability(team1_name, team2_name, neutral=True)
    
    # Predict with covariates
    # We use theta_a = 0.15 and theta_d = 0.04 to scale the covariates appropriately
    adjusted_pred = model.predict_probability(
        team1_name, team2_name, 
        neutral=True, 
        home_covariates=team1_covs, 
        away_covariates=team2_covs,
        theta_a=0.15,
        theta_d=0.04
    )
    
    print("\n==================================================")
    print("      MATCH PREDICTION REPORT")
    print("==================================================")
    
    print("\n1. BASELINE MODEL (Historical Results Only):")
    print(f"  Expected Goals: {team1_name} {baseline_pred['lambda']:.2f} - {baseline_pred['mu']:.2f} {team2_name}")
    print(f"  {team1_name} Win:      {baseline_pred['home_win']*100:.2f}%")
    print(f"  Draw:            {baseline_pred['draw']*100:.2f}%")
    print(f"  {team2_name} Win:   {baseline_pred['away_win']*100:.2f}%")
    
    print("\n2. ADJUSTED MODEL (Incorporating 1/11 Starting Lineup Form):")
    print(f"  Expected Goals: {team1_name} {adjusted_pred['lambda']:.2f} - {adjusted_pred['mu']:.2f} {team2_name}")
    print(f"  {team1_name} Win:      {adjusted_pred['home_win']*100:.2f}%")
    print(f"  Draw:            {adjusted_pred['draw']*100:.2f}%")
    print(f"  {team2_name} Win:   {adjusted_pred['away_win']*100:.2f}%")
    
    print_exact_scores(adjusted_pred["matrix"], team1_name, team2_name)
    
    # Save predictions to output
    pred_path = Path(__file__).parent / "output" / "match_prediction.json"
    # Save serializable part of prediction
    output_pred = {
        "baseline": {
            "lambda": baseline_pred["lambda"],
            "mu": baseline_pred["mu"],
            "home_win": baseline_pred["home_win"],
            "draw": baseline_pred["draw"],
            "away_win": baseline_pred["away_win"]
        },
        "adjusted": {
            "lambda": adjusted_pred["lambda"],
            "mu": adjusted_pred["mu"],
            "home_win": adjusted_pred["home_win"],
            "draw": adjusted_pred["draw"],
            "away_win": adjusted_pred["away_win"]
        }
    }
    pred_path.write_text(json.dumps(output_pred, indent=2))
    print(f"\nSaved prediction results to {pred_path}")
    print("==================================================")

if __name__ == "__main__":
    main()
