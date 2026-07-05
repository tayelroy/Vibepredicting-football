import os
import json
import pandas as pd
import sys
from pathlib import Path
# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from pipeline.sportradar_pipeline import SporTradarClient, discover_national_team_ids, build_intl_results_df

def main():
    api_key = os.getenv("sportradar_api_key")
    if not api_key:
        raise EnvironmentError("sportradar_api_key not found.")
        
    client = SporTradarClient(api_key=api_key)
    
    print("Discovering national team IDs for all 14 target nations...")
    national_team_ids = discover_national_team_ids(client)
    print(f"Discovered {len(national_team_ids)} / 14 national team IDs:")
    print(json.dumps(national_team_ids, indent=2))
    
    # Save nation IDs to output
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    nation_ids_path = output_dir / "national_team_ids.json"
    nation_ids_path.write_text(json.dumps(national_team_ids, indent=2, ensure_ascii=False))
    
    # Fetch summaries for all discovered nations
    print("\nFetching match summaries for national teams...")
    nation_summaries = {}
    for nation, team_id in national_team_ids.items():
        print(f"  Fetching summaries for {nation} ({team_id})...")
        try:
            summaries_data = client.competitor_summaries(team_id)
            # Cache it using client cache path
            endpoint = f"/competitors/{team_id}/summaries.json"
            client._write_cache(endpoint, summaries_data)
            
            nation_summaries[nation] = summaries_data.get("summaries", [])
        except Exception as e:
            print(f"  ❌ Error fetching summaries for {nation}: {e}")
            
    # Flatten and build results DataFrame
    print("\nBuilding international results DataFrame...")
    df_intl = build_intl_results_df(nation_summaries)
    
    csv_path = output_dir / "intl_results.csv"
    df_intl.to_csv(csv_path, index=False)
    print(f"Saved {len(df_intl)} completed matches to {csv_path}")

if __name__ == "__main__":
    main()
