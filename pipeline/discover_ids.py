import os
import json
import re
import unicodedata
import sys
from pathlib import Path
# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from pipeline.sportradar_pipeline import SporTradarClient, TARGET_SEASON_YEARS

# Mapping of league names to their exact Sportradar competition IDs
LEAGUE_COMPETITIONS = {
    "Premier League": "sr:competition:17",
    "La Liga": "sr:competition:8",
    "Primeira Liga": "sr:competition:238",
    "Ligue 1": "sr:competition:34",
    "Saudi Pro League": "sr:competition:955",
    "Serie A": "sr:competition:23",
    "Bundesliga": "sr:competition:9",
    "Eredivisie": "sr:competition:39",
    "Championship": "sr:competition:18",
    "Brasileirão": "sr:competition:325",
    "Eliteserien": "sr:competition:116"
}

# Manual competitor club ID mappings for quick resolution and unsupported API leagues
MANUAL_CLUB_IDS = {
    "Zenit Saint Petersburg": "sr:competitor:1946",
    "AZ Alkmaar": "sr:competitor:2956",
    "Bodø/Glimt": "sr:competitor:2054",
    "RB Leipzig": "sr:competitor:2807",
    "Nottingham Forest": "sr:competitor:60"
}

# Strict mappings to prevent substring mismatch (e.g. Barcelona matching Espanyol Barcelona first)
MANUAL_CLUB_MATCHES = {
    "barcelona": "fcbarcelona",
    "athleticclub": "athleticbilbao",
    "realsociedad": "realsociedadsansebastian",
    "atleticomadrid": "atleticomadrid",
    "alhilal": "alhilalsfc",
    "alnassr": "alnassrclub",
    "psg": "parissaintgermain",
    "fcporto": "fcporto",
    "sportingcp": "sportingcp",
    "tottenhamhotspur": "tottenhamhotspur",
    "manchestercity": "manchestercity",
    "manchesterunited": "manchesterunited",
    "chelsea": "chelseafc",
    "juventus": "juventusturin",
    "acmilan": "acmilan",
    "liverpool": "liverpoolfc",
    "arsenal": "arsenalfc",
    "sevilla": "sevillafc",
    "torino": "torinofc",
    "brentford": "brentfordfc",
    "westbromwichalbion": "westbromwichalbion",
    "fulham": "fulhamfc"
}

def normalize_name(s):
    s = s.lower()
    # Handle Scandinavian characters that don't normalize with NFD
    s = s.replace('ø', 'o').replace('æ', 'ae').replace('å', 'aa')
    s = "".join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    s = re.sub(r'[^a-z0-9]', '', s)
    return s

def name_match(target_club, candidate_club):
    t_norm = normalize_name(target_club)
    c_norm = normalize_name(candidate_club)
    
    # Use strict manual matching if defined
    if t_norm in MANUAL_CLUB_MATCHES:
        return MANUAL_CLUB_MATCHES[t_norm] == c_norm
        
    return t_norm == c_norm or t_norm in c_norm or c_norm in t_norm

def player_name_match(name1, name2):
    def get_parts(s):
        s = s.lower()
        # Handle Scandinavian characters in player names too
        s = s.replace('ø', 'o').replace('æ', 'ae').replace('å', 'aa')
        s = "".join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
        parts = re.split(r'[\s,]+', s)
        return [p for p in parts if p]
    
    parts1 = get_parts(name1)
    parts2 = get_parts(name2)
    if not parts1 or not parts2:
        return False
    
    # Check if all parts of one name are present in the other (e.g. Lamine Yamal -> Yamal, Lamine)
    if all(any(p1 in p2 or p2 in p1 for p2 in parts2) for p1 in parts1):
        return True
    if all(any(p2 in p1 or p1 in p2 for p1 in parts1) for p2 in parts2):
        return True
    return False

def discover_club_ids(client, unique_clubs):
    print("\n--- Discovering Club IDs ---")
    club_ids = {}
    
    # First apply manual club IDs
    for target_club in unique_clubs:
        if target_club in MANUAL_CLUB_IDS:
            club_ids[target_club] = MANUAL_CLUB_IDS[target_club]
            print(f"  Mapped Club (manual): {target_club} -> {MANUAL_CLUB_IDS[target_club]}")
            
    # Discover remaining clubs
    remaining_clubs = [c for c in unique_clubs if c not in club_ids]
    if not remaining_clubs:
        return club_ids
        
    for comp_name, comp_id in LEAGUE_COMPETITIONS.items():
        print(f"Checking league: {comp_name} ({comp_id})...")
        try:
            seasons_data = client.competition_seasons(comp_id)
            seasons = seasons_data.get("seasons", [])
            # Find the most recent season
            relevant_seasons = [
                s["id"] for s in seasons
                if any(t in s.get("year", "") for t in TARGET_SEASON_YEARS)
            ]
            if not relevant_seasons and seasons:
                relevant_seasons = [seasons[0]["id"]]
                
            if not relevant_seasons:
                continue
                
            season_id = relevant_seasons[0]
            
            # Fetch teams in that season
            sc_data = client.season_competitors(season_id)
            competitors = sc_data.get("season_competitors", sc_data.get("competitors", []))
            
            for c in competitors:
                c_name = c.get("name", "")
                c_id = c.get("id", "")
                
                for target_club in remaining_clubs:
                    if target_club not in club_ids and name_match(target_club, c_name):
                        club_ids[target_club] = c_id
                        print(f"  Mapped Club: {target_club} -> {c_name} ({c_id})")
        except Exception as e:
            print(f"  Error processing league {comp_name}: {e}")
            
    # Check if any clubs are missing
    missing = [c for c in unique_clubs if c not in club_ids]
    if missing:
        print(f"⚠️ Warning: Could not find IDs for clubs: {missing}")
                
    return club_ids

def get_national_team_ids(client, target_nations):
    print("\n--- Discovering National Team IDs ---")
    
    # Target countries mapping
    iso_codes = {
        "Spain": "ESP", "Portugal": "POR", "Brazil": "BRA", "Norway": "NOR",
        "France": "FRA", "England": "ENG", "Belgium": "BEL", "Germany": "GER",
        "Argentina": "ARG", "Colombia": "COL", "USA": "USA", "Egypt": "EGY",
        "Morocco": "MAR", "Mexico": "MEX", "Switzerland": "SUI"
    }
    targets = {nation: iso_codes.get(nation, "") for nation in target_nations}
    team_ids = {}
    
    competitors = []
    # Try World Cup first
    try:
        wc_seasons = client.competition_seasons("sr:competition:16").get("seasons", [])
        if wc_seasons:
            season_id = wc_seasons[0]["id"]
            sc_data = client.season_competitors(season_id)
            competitors.extend(sc_data.get("season_competitors", sc_data.get("competitors", [])))
    except Exception as e:
        print(f"  Error loading World Cup competitors: {e}")
        
    # Try Nations League as fallback
    try:
        unl_seasons = client.competition_seasons("sr:competition:23755").get("seasons", [])
        if unl_seasons:
            season_id = unl_seasons[0]["id"]
            sc_data = client.season_competitors(season_id)
            competitors.extend(sc_data.get("season_competitors", sc_data.get("competitors", [])))
    except Exception as e:
        print(f"  Error loading Nations League competitors: {e}")
        
    # Map target nations
    for c in competitors:
        c_name = c.get("name", "")
        c_code = c.get("country_code", "").upper()
        for nation, iso in targets.items():
            if nation in team_ids:
                continue
            if (iso and c_code == iso) or name_match(nation, c_name):
                team_ids[nation] = c["id"]
                print(f"  Mapped Nation: {nation} -> {c_name} ({c['id']})")
                
    return team_ids

def discover_player_ids(client, roster, national_team_ids, club_ids):
    print("\n--- Discovering Player IDs ---")
    player_ids = {}
    
    # 1. Scan national team lineups dynamically
    for nation, team_id in national_team_ids.items():
        # Map back to roster key (e.g. Spain -> spain_xi)
        key = f"{nation.lower().replace(' ', '_')}_xi"
        target_players = roster.get(key, [])
        
        print(f"Scanning match lineups for {nation}...")
        try:
            summaries_data = client.competitor_summaries(team_id)
            summaries = summaries_data.get("summaries", [])
            
            completed = [
                s for s in summaries
                if s.get("sport_event_status", {}).get("status") in ("closed", "ended")
            ][:5] # Scan last 5 matches
            
            seen_players = {} # name -> id
            for match in completed:
                match_id = match.get("sport_event", {}).get("id", "")
                if not match_id:
                    continue
                try:
                    lineups_data = client.sport_event_lineups(match_id)
                    competitors = lineups_data.get("lineups", {}).get("competitors", [])
                    for comp in competitors:
                        if comp.get("id") == team_id:
                            for p in comp.get("players", []):
                                seen_players[p.get("name")] = p.get("id")
                except Exception as e:
                    print(f"  Error reading lineups for match {match_id}: {e}")
                    
            # Match target players
            for tp in target_players:
                tp_name = tp["player_name"]
                for s_name, s_id in seen_players.items():
                    if player_name_match(tp_name, s_name):
                        player_ids[tp_name] = s_id
                        print(f"  Mapped Player: {tp_name} -> {s_name} ({s_id})")
                        break
        except Exception as e:
            print(f"  Error mapping via national team summaries for {nation}: {e}")
                    
    # 2. For any missing players, look them up in their club's player list
    xi_keys = [k for k in roster.keys() if k.endswith("_xi")]
    all_targets = []
    for k in xi_keys:
        all_targets.extend(roster[k])
        
    missing_players = [p for p in all_targets if p["player_name"] not in player_ids]
    
    if missing_players:
        print(f"\nScanning club squads for {len(missing_players)} missing players...")
        for p in missing_players:
            p_name = p["player_name"]
            p_club = p["club"]
            club_id = club_ids.get(p_club)
            if not club_id:
                print(f"  Cannot scan club for {p_name} because {p_club} ID is missing.")
                continue
                
            # Search in club's summaries (reused and cached)
            print(f"  Searching for {p_name} in {p_club} summaries...")
            try:
                summaries_data = client.competitor_summaries(club_id)
                
                # Cache it using client cache path
                endpoint = f"/competitors/{club_id}/summaries.json"
                client._write_cache(endpoint, summaries_data)
                
                club_seen_players = {}
                for match in summaries_data.get("summaries", [])[:5]:
                    stats = match.get("statistics")
                    if stats:
                        for comp in stats.get("totals", {}).get("competitors", []):
                            if comp.get("id") == club_id:
                                for pl in comp.get("players", []):
                                    club_seen_players[pl.get("name")] = pl.get("id")
                                    
                for s_name, s_id in club_seen_players.items():
                    if player_name_match(p_name, s_name):
                        player_ids[p_name] = s_id
                        print(f"  Mapped Player (club): {p_name} -> {s_name} ({s_id})")
                        break
            except Exception as e:
                print(f"  Error reading club summaries for {p_club}: {e}")
                
    # Final check
    still_missing = [p["player_name"] for p in all_targets if p["player_name"] not in player_ids]
    if still_missing:
        print(f"⚠️ Still missing player IDs for: {still_missing}")
        # Add manual mappings for any players that could not be found
        manual_player_ids = {
            "Alex Baena": "sr:player:1340446",
        }
        for sm in still_missing:
            if sm in manual_player_ids:
                player_ids[sm] = manual_player_ids[sm]
                print(f"  Mapped Player (manual): {sm} -> {manual_player_ids[sm]}")
                
    return player_ids

def main():
    api_key = os.getenv("sportradar_api_key")
    if not api_key:
        raise EnvironmentError("sportradar_api_key not found.")
        
    client = SporTradarClient(api_key=api_key)
    
    # Load roster
    roster_path = Path(__file__).parent.parent / "roster.json"
    if not roster_path.exists():
        raise FileNotFoundError("roster.json not found.")
    roster = json.loads(roster_path.read_text())
    
    # Get dynamic lineup keys
    xi_keys = [k for k in roster.keys() if k.endswith("_xi")]
    
    # Gather target players
    all_players = []
    for k in xi_keys:
        all_players.extend(roster[k])
        
    # Get unique clubs
    unique_clubs = sorted(list(set(p["club"] for p in all_players)))
    print(f"Unique clubs ({len(unique_clubs)}): {unique_clubs}")
    
    # Discover club IDs
    club_ids = discover_club_ids(client, unique_clubs)
    
    # Discover national team IDs
    target_nations = [k[:-3].replace("_", " ").title() for k in xi_keys]
    target_nations = ["USA" if n.lower() == "usa" else n for n in target_nations]
    
    national_team_ids = get_national_team_ids(client, target_nations)
    
    # Discover player IDs
    player_ids = discover_player_ids(client, roster, national_team_ids, club_ids)
    
    # Save mappings
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    mappings = {
        "clubs": club_ids,
        "nations": national_team_ids,
        "players": player_ids
    }
    
    output_path = output_dir / "roster_ids.json"
    output_path.write_text(json.dumps(mappings, indent=2, ensure_ascii=False))
    print(f"\nSaved ID mappings to {output_path}")

if __name__ == "__main__":
    main()
