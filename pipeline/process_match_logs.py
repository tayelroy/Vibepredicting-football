import os
import json
import pandas as pd
import sys
from pathlib import Path
# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from pipeline.sportradar_pipeline import SporTradarClient

def load_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text())

def calculate_player_average_form(client, player_id, club_id, nation_id, player_name, min_matches=8, min_minutes=15):
    """
    Fetches the summaries for the player's club, finds the last 'min_matches' matches
    where the player played at least 'min_minutes', and averages their per-90 stats.
    Falls back to national team summaries if club summaries have no matches.
    """
    print(f"\nProcessing player: {player_name} ({player_id})")
    
    # Try club first
    player_match_stats = []
    
    def scan_summaries(target_team_id, label):
        print(f"  Scanning {label} summaries ({target_team_id})...")
        try:
            summaries_data = client.competitor_summaries(target_team_id)
            
            # Cache it
            endpoint = f"/competitors/{target_team_id}/summaries.json"
            client._write_cache(endpoint, summaries_data)
            
            summaries = summaries_data.get("summaries", [])
            matches = []
            
            for s in summaries:
                status = s.get("sport_event_status", {})
                if status.get("status") not in ("closed", "ended"):
                    continue
                    
                stats = s.get("statistics")
                if not stats:
                    continue
                    
                competitors = stats.get("totals", {}).get("competitors", [])
                
                # Find the competitor entry matching our target team ID
                our_idx = next((i for i, c in enumerate(competitors) if c.get("id") == target_team_id), None)
                if our_idx is None:
                    continue
                    
                # Opponent index
                opp_idx = 1 - our_idx if len(competitors) == 2 else None
                
                # Find our player in the player list
                players = competitors[our_idx].get("players", [])
                player_entry = next((p for p in players if p.get("id") == player_id), None)
                if not player_entry:
                    continue
                    
                p_stats = player_entry.get("statistics", {})
                
                # Handle missing minutes_played
                minutes = p_stats.get("minutes_played")
                if minutes is None:
                    # Estimate based on starting/sub status
                    starter = player_entry.get("starter", False)
                    sub_in = p_stats.get("substituted_in", 0)
                    sub_out = p_stats.get("substituted_out", 0)
                    if starter:
                        minutes = 70 if sub_out == 1 else 90
                    else:
                        minutes = 20 if sub_in == 1 else 0
                else:
                    minutes = float(minutes)
                    
                if minutes >= min_minutes:
                    # Calculate team-level PPDA and Shot Suppression for the match
                    our_team_stats = competitors[our_idx].get("statistics", {})
                    opp_team_stats = competitors[opp_idx].get("statistics", {}) if opp_idx is not None else {}
                    
                    # Our team defensive actions
                    our_tackles = float(our_team_stats.get("tackles_total", 0) or 0)
                    our_interceptions = float(our_team_stats.get("interceptions", 0) or 0)
                    our_clearances = float(our_team_stats.get("clearances", 0) or 0)
                    our_blocks = float(our_team_stats.get("defensive_blocks", 0) or 0)
                    our_defensive_actions = our_tackles + our_interceptions + our_clearances + our_blocks
                    
                    # Opponent passes total
                    opp_passes = float(opp_team_stats.get("passes_total", 0) or 0)
                    match_ppda = opp_passes / max(our_defensive_actions, 1)
                    
                    # Opponent shots total (Shot suppression targets)
                    opp_shots = float(opp_team_stats.get("shots_total", 0) or 0)
                    opp_shots_on_target = float(opp_team_stats.get("shots_on_target", 0) or 0)
                    
                    matches.append({
                        "date": s.get("sport_event", {}).get("start_time", "")[:10],
                        "match_id": s.get("sport_event", {}).get("id"),
                        "minutes_played": minutes,
                        "stats": p_stats,
                        "team_ppda": match_ppda,
                        "team_shots_conceded": opp_shots,
                        "team_shots_on_target_conceded": opp_shots_on_target
                    })
            return matches
        except Exception as e:
            print(f"  ❌ Error reading summaries for {label}: {e}")
            return []
            
    # Step A: Scan Club Summaries
    if club_id:
        player_match_stats = scan_summaries(club_id, "club")
        
    # Step B: Fallback to National Team Summaries if no club matches found
    if not player_match_stats and nation_id:
        print(f"  ⚠️ No matches found in club summaries. Falling back to national team...")
        player_match_stats = scan_summaries(nation_id, "national team")
        
    if not player_match_stats:
        print(f"  ❌ No matches found anywhere where {player_name} played at least {min_minutes} minutes.")
        return None
        
    # Sort by date descending
    player_match_stats = sorted(player_match_stats, key=lambda x: x["date"], reverse=True)
    
    # Keep the last 'min_matches' matches
    target_matches = player_match_stats[:min_matches]
    print(f"  Found {len(player_match_stats)} matches. Using {len(target_matches)} recent matches.")
    
    # Accumulate per-90 rates
    all_metrics = []
    for m in target_matches:
        minutes = m["minutes_played"]
        raw = m["stats"]
        
        # Extract raw metrics (with fallback to 0)
        goals = float(raw.get("goals_scored", 0) or 0)
        penalties = float(raw.get("goals_by_penalty", 0) or 0)
        shots_on_target = float(raw.get("shots_on_target", 0) or 0)
        shots_off_target = float(raw.get("shots_off_target", 0) or 0)
        chances = float(raw.get("chances_created", 0) or 0)
        dribbles = float(raw.get("dribbles_completed", 0) or 0)
        crosses_succ = float(raw.get("crosses_successful", 0) or 0)
        long_passes_succ = float(raw.get("long_passes_successful", 0) or 0)
        tackles_succ = float(raw.get("tackles_successful", 0) or 0)
        interceptions = float(raw.get("interceptions", 0) or 0)
        clearances = float(raw.get("clearances", 0) or 0)
        blocks = float(raw.get("defensive_blocks", 0) or 0)
        fouls = float(raw.get("fouls_committed", 0) or 0)
        passes = float(raw.get("passes_total", 0) or 0)
        shots_faced = float(raw.get("shots_faced", 0) or 0)
        
        # Normalize to per-90
        factor = 90.0 / minutes
        
        all_metrics.append({
            "npxG_raw": ((shots_on_target * 0.30) + (shots_off_target * 0.05)) * factor,
            "chances_created_raw": chances * factor,
            "dribbles_completed_raw": dribbles * factor,
            "crosses_succ_raw": crosses_succ * factor,
            "long_passes_succ_raw": long_passes_succ * factor,
            "tackles_succ_raw": tackles_succ * factor,
            "interceptions_raw": interceptions * factor,
            "clearances_raw": clearances * factor,
            "blocks_raw": blocks * factor,
            "fouls_raw": fouls * factor,
            "passes_total_raw": passes * factor,
            "shots_faced_raw": shots_faced * factor,
            "team_ppda": m["team_ppda"],
            "team_shots_conceded": m["team_shots_conceded"],
            "team_shots_on_target_conceded": m["team_shots_on_target_conceded"]
        })
        
    # Calculate mean of per-90 metrics
    df = pd.DataFrame(all_metrics)
    mean_metrics = df.mean().to_dict()
    
    # Structure the final profile
    profile = {
        "player_name": player_name,
        "player_id": player_id,
        "club_id": club_id,
        "matches_analyzed": len(target_matches),
        "npxG_per90": mean_metrics["npxG_raw"],
        "chances_created_per90": mean_metrics["chances_created_raw"],
        "dribbles_completed_per90": mean_metrics["dribbles_completed_raw"],
        "crosses_succ_per90": mean_metrics["crosses_succ_raw"],
        "long_passes_succ_per90": mean_metrics["long_passes_succ_raw"],
        "tackles_succ_per90": mean_metrics["tackles_succ_raw"],
        "interceptions_per90": mean_metrics["interceptions_raw"],
        "clearances_per90": mean_metrics["clearances_raw"],
        "blocks_per90": mean_metrics["blocks_raw"],
        "fouls_per90": mean_metrics["fouls_raw"],
        "passes_total_per90": mean_metrics["passes_total_raw"],
        "shots_faced_per90": mean_metrics["shots_faced_raw"],
        "team_ppda_average": mean_metrics["team_ppda"],
        "team_shots_conceded_average": mean_metrics["team_shots_conceded"],
        "team_shots_on_target_conceded_average": mean_metrics["team_shots_on_target_conceded"]
    }
    
    return profile

def main():
    api_key = os.getenv("sportradar_api_key")
    if not api_key:
        raise EnvironmentError("sportradar_api_key not found.")
        
    client = SporTradarClient(api_key=api_key)
    
    # Load roster and IDs
    roster = load_json(Path(__file__).parent.parent / "roster.json")
    ids = load_json(Path(__file__).parent.parent / "output" / "roster_ids.json")
    
    if not roster or not ids:
        print("Error: roster.json or roster_ids.json missing. Run discover_ids.py first.")
        return
        
    club_ids = ids.get("clubs", {})
    player_ids = ids.get("players", {})
    nation_ids = ids.get("nations", {})
    
    player_profiles = {}
    csv_rows = []
    
    for nation_key in ["spain_xi", "portugal_xi"]:
        nation_name = "Spain" if nation_key == "spain_xi" else "Portugal"
        player_profiles[nation_name] = []
        
        n_id = nation_ids.get(nation_name)
        
        for p in roster.get(nation_key, []):
            name = p["player_name"]
            club = p["club"]
            
            p_id = player_ids.get(name)
            c_id = club_ids.get(club)
            
            if not p_id:
                print(f"⚠️ Missing player ID for {name}. Skipping.")
                continue
                
            profile = calculate_player_average_form(client, p_id, c_id, n_id, name)
            if profile:
                profile["role"] = p["role"]
                profile["club_name"] = club
                profile["nation"] = nation_name
                player_profiles[nation_name].append(profile)
                
                # Create a row for CSV
                csv_rows.append({
                    "nation": nation_name,
                    "player_name": name,
                    "club_name": club,
                    "role": p["role"],
                    "npxG_per90": round(profile["npxG_per90"], 4),
                    "chances_created_per90": round(profile["chances_created_per90"], 4),
                    "dribbles_completed_per90": round(profile["dribbles_completed_per90"], 4),
                    "crosses_succ_per90": round(profile["crosses_succ_per90"], 4),
                    "long_passes_succ_per90": round(profile["long_passes_succ_per90"], 4),
                    "tackles_succ_per90": round(profile["tackles_succ_per90"], 4),
                    "interceptions_per90": round(profile["interceptions_per90"], 4),
                    "clearances_per90": round(profile["clearances_per90"], 4),
                    "blocks_per90": round(profile["blocks_per90"], 4),
                    "fouls_per90": round(profile["fouls_per90"], 4),
                    "passes_total_per90": round(profile["passes_total_per90"], 4),
                    "shots_faced_per90": round(profile["shots_faced_per90"], 4),
                    "team_ppda_average": round(profile["team_ppda_average"], 4),
                    "team_shots_conceded_average": round(profile["team_shots_conceded_average"], 4)
                })
                
    # Save profiles
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "player_profiles.json"
    output_path.write_text(json.dumps(player_profiles, indent=2, ensure_ascii=False))
    print(f"\nSaved player profiles to {output_path}")
    
    # Save CSV
    df = pd.DataFrame(csv_rows)
    csv_path = output_dir / "player_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved player metrics to {csv_path}")

if __name__ == "__main__":
    main()
