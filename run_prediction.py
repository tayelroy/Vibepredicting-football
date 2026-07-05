import json
import subprocess
import sys
from pathlib import Path

def load_json(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text())

def run_script(script_path):
    print(f"\n Running: {script_path}")
    result = subprocess.run([sys.executable, str(script_path)], capture_output=False)
    if result.returncode != 0:
        print(f"Error: {script_path} failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def _is_cache_invalid(has_results, teams, valid_ids):
    # Evaluates if the local dataset requires a fresh API pull 
    if not has_results:
        return True
    return any(team not in valid_ids for team in teams)

def main():
    print("==================================================")
    print("      WC 2026 PREDICTION ENGINE RUNNER")
    print("==================================================")
    
    # 1. Parse active teams in roster.json
    roster_path = Path("roster.json")
    if not roster_path.exists():
        print("Error: roster.json not found in root directory.")
        sys.exit(1)
        
    roster = load_json(roster_path)
    xi_keys = [k for k in roster.keys() if k.endswith("_xi")]
    
    if len(xi_keys) < 2:
        print("Error: roster.json must contain at least 2 team lineups (e.g. spain_xi, portugal_xi).")
        sys.exit(1)
        
    # Dynamically extract all active team names
    teams = [key[:-3].replace("_", " ").title() for key in xi_keys]
    
    print(f"Active Matchup : {' vs. '.join(teams)}")
    
    # 2. Check if we have international results compiled for these teams
    nation_ids = load_json(Path("output/national_team_ids.json"))
    has_results = Path("output/intl_results.csv").exists()
    
    need_intl_rebuild = _is_cache_invalid(has_results, teams, nation_ids) #Check if we need another API call
    
    if need_intl_rebuild:
        missing_teams = [t for t in teams if t not in nation_ids]
        if missing_teams:
            print(f" International match database does not contain: {', '.join(missing_teams)}.")
            # Read docs to find the nationality code
        else:
            print("⚠️ International match database is missing.")
        print("Compiling international match database first...")
        run_script(Path("pipeline/build_intl_results.py"))
        
    # 3. Execute the pipeline steps
    run_script(Path("pipeline/discover_ids.py"))
    run_script(Path("pipeline/process_match_logs.py"))
    run_script(Path("aggregation/aggregate_national_teams.py"))
    run_script(Path("predict.py"))
    
    print("\n✅ Prediction run completed successfully!")

if __name__ == "__main__":
    main()
