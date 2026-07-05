import json
import pandas as pd
import sys
from pathlib import Path
# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from aggregation.module_synthetic_aggregation import (
    POSITIONAL_WEIGHTS,
    LEAGUE_COEFFICIENTS,
    _resolve_role,
    _resolve_league_coefficient
)

def load_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text())

def aggregate_team_profile(team_name, roster, player_profiles):
    """
    Aggregates the player-level match log averages using the two-stream decoupled formula.
    """
    print(f"\nAggregating team profile for: {team_name}")
    
    # Track team totals
    team_npxG = 0.0
    team_xT = 0.0
    team_Deep_Progressions = 0.0
    team_PPDA = 0.0
    team_Shot_Suppression = 0.0
    
    # Store individual player details for logging/diagnostics
    player_details = []
    
    for p in roster:
        name = p["player_name"]
        role_raw = p["role"]
        role = _resolve_role(role_raw)
        
        # Find player profile
        profile = next((pl for pl in player_profiles if pl["player_name"] == name), None)
        if not profile:
            print(f"  ❌ Missing profile for player {name}. Skipping contribution.")
            continue
            
        league_coeff = _resolve_league_coefficient(p["league"])
        
        # --- Calculate player metrics ---
        npxG_p90 = profile["npxG_per90"]
        chances_p90 = profile["chances_created_per90"]
        dribbles_p90 = profile["dribbles_completed_per90"]
        crosses_p90 = profile["crosses_succ_per90"]
        long_passes_p90 = profile["long_passes_succ_per90"]
        
        # xT Proxy = chances*0.4 + dribbles*0.3 + crosses*0.2 + long_passes*0.1
        xT_p90 = chances_p90 * 0.4 + dribbles_p90 * 0.3 + crosses_p90 * 0.2 + long_passes_p90 * 0.1
        
        # Deep Progressions = crosses + chances
        deep_prog_p90 = crosses_p90 + chances_p90
        
        # Systemic match-averages
        team_ppda_avg = profile["team_ppda_average"]
        team_shots_avg = profile["team_shots_conceded_average"]
        
        # --- Stream A: Additive Metrics ---
        npxG_contrib = npxG_p90 * league_coeff
        xT_contrib = xT_p90 * league_coeff
        deep_prog_contrib = deep_prog_p90 * league_coeff
        
        # --- Stream B: Systemic Metrics (Scaled by Positional Weights) ---
        ppda_weight = POSITIONAL_WEIGHTS["PPDA"].get(role, 0.0)
        ppda_contrib = team_ppda_avg * ppda_weight * league_coeff
        
        shot_supp_weight = POSITIONAL_WEIGHTS["Shot_Suppression"].get(role, 0.0)
        shot_supp_contrib = team_shots_avg * shot_supp_weight * league_coeff
        
        # Accumulate
        team_npxG += npxG_contrib
        team_xT += xT_contrib
        team_Deep_Progressions += deep_prog_contrib
        team_PPDA += ppda_contrib
        team_Shot_Suppression += shot_supp_contrib
        
        player_details.append({
            "player_name": name,
            "role": role,
            "league_coeff": league_coeff,
            "npxG_contrib": npxG_contrib,
            "xT_contrib": xT_contrib,
            "deep_prog_contrib": deep_prog_contrib,
            "ppda_contrib": ppda_contrib,
            "shot_supp_contrib": shot_supp_contrib
        })
        
        print(f"  {name} ({role}):")
        print(f"    Stream A -> npxG_contrib: {npxG_contrib:.4f} | xT_contrib: {xT_contrib:.4f} | deep_prog_contrib: {deep_prog_contrib:.4f}")
        print(f"    Stream B -> ppda_contrib: {ppda_contrib:.4f} (wt={ppda_weight:.2f}) | shot_supp_contrib: {shot_supp_contrib:.4f} (wt={shot_supp_weight:.2f})")
        
    print(f"\nSynthetic Profile for {team_name}:")
    print(f"  npxG:              {team_npxG:.4f}")
    print(f"  xT_Proxy:          {team_xT:.4f}")
    print(f"  Deep_Progressions: {team_Deep_Progressions:.4f}")
    print(f"  PPDA:              {team_PPDA:.4f}")
    print(f"  Shot_Suppression:  {team_Shot_Suppression:.4f}")
    
    return {
        "team": team_name,
        "npxG": team_npxG,
        "xT_Proxy": team_xT,
        "Deep_Progressions": team_Deep_Progressions,
        "PPDA": team_PPDA,
        "Shot_Suppression": team_Shot_Suppression
    }, player_details

def main():
    roster = load_json(Path(__file__).parent.parent / "roster.json")
    profiles = load_json(Path(__file__).parent.parent / "output" / "player_profiles.json")
    
    if not roster or not profiles:
        print("Error: roster.json or player_profiles.json missing. Run previous steps first.")
        return
        
    xi_keys = [k for k in roster.keys() if k.endswith("_xi")]
    team_profiles = []
    details_map = {}
    
    for key in xi_keys:
        nation_name = key[:-3].replace("_", " ").title()
        if nation_name.lower() == "usa":
            nation_name = "USA"
        profile, details = aggregate_team_profile(
            nation_name, roster[key], profiles.get(nation_name, [])
        )
        team_profiles.append(profile)
        details_map[nation_name] = details
        
    # Save synthesized team profiles
    df_teams = pd.DataFrame(team_profiles)
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    csv_path = output_dir / "synthetic_national_teams.csv"
    df_teams.to_csv(csv_path, index=False)
    print(f"\nSaved synthetic team profiles to {csv_path}")
    
    # Save details as JSON
    details_path = output_dir / "synthetic_aggregation_details.json"
    details_path.write_text(json.dumps(details_map, indent=2, ensure_ascii=False))
    print(f"Saved detailed synthetic contributions to {details_path}")

if __name__ == "__main__":
    main()
