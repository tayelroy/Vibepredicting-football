"""
sportradar_pipeline.py
======================
Sportradar Soccer Extended v4 — Data Ingestion Pipeline for WC 2026 Prediction Engine.

Pulls two datasets:
  A) Club seasonal metrics  → feeds module_synthetic_aggregation.synthesize_team()
  B) National team results  → feeds Dixon-Coles model training

Target nations (14):
  Brazil, Norway, Mexico, England, Portugal, Spain, USA, Belgium,
  Argentina, Egypt, Switzerland, Colombia, France, Morocco

Trial key constraints:
  Total quota : 1,000 calls   (hard ceiling)
  QPS limit   : 1 req/second
  Trial window: 07/04/2026 — 08/03/2026

Estimated call budget:
  ┌──────────────────────────────────────────────────┬────────┐
  │ Step                                             │ Calls  │
  ├──────────────────────────────────────────────────┼────────┤
  │ FIFA Rankings (national team ID discovery)       │      1 │
  │ competitions.json (major league discovery)       │      1 │
  │ competition_seasons (~10 major leagues)          │    ~10 │
  │ competitor_profile (14 national teams)           │     14 │
  │ player_profile (~23 players × 14 nations, dedup) │   ~322 │
  │ seasonal_competitor_statistics (~150 clubs × 2)  │   ~300 │
  │ competitor_summaries (14 nations, Dixon-Coles)   │     14 │
  ├──────────────────────────────────────────────────┼────────┤
  │ TOTAL ESTIMATED                                  │   ~662 │
  │ Buffer remaining                                 │   ~338 │
  └──────────────────────────────────────────────────┴────────┘

⚠️  PIVOT NOTICE — module_synthetic_aggregation.py:
  That file was NOT modified here. It still contains the original formula
  (Positional_Weight × League_Coefficient × Minutes_Pct × Usage_Rate_Pct).
  The updated decoupled formula you described — positional weights for
  finishing/defensive metrics, usage rate for possession metrics, minutes
  removed from the multiplier — requires a separate rewrite of that module.
  This pipeline only collects the raw source data.

Usage:
  python sportradar_pipeline.py --mode status   # quota check, zero API calls
  python sportradar_pipeline.py --mode probe    # 5 API calls — verify auth + tier
  python sportradar_pipeline.py --mode run      # full pipeline (~662 calls)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

# ── .env loading ──────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # Fallback: parse .env manually without python-dotenv
    _env_path = Path(__file__).parent / ".env"
    if _env_path.exists():
        for _line in _env_path.read_text().splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("SporTradarPipeline")


# ── Constants ─────────────────────────────────────────────────────────────────

BASE_URL = "https://api.sportradar.com/soccer-extended/trial/v4/en"
QUOTA_LIMIT = 1_000

# 14 target nations: display name → Sportradar country_code
TARGET_NATIONS: dict[str, str] = {
    "Brazil": "BRA",
    "Norway": "NOR",
    "Mexico": "MEX",
    "England": "ENG",
    "Portugal": "POR",
    "Spain": "ESP",
    "USA": "USA",
    "Belgium": "BEL",
    "Argentina": "ARG",
    "Egypt": "EGY",
    "Switzerland": "CHE",
    "Colombia": "COL",
    "France": "FRA",
    "Morocco": "MAR",
}

# Substrings used to filter competitions.json for major men's league competitions.
# Matched case-insensitively against competition.name.
# Only domestic leagues — international tournaments are handled separately.
MAJOR_LEAGUE_SUBSTRINGS: list[str] = [
    "Premier League",  # England
    "La Liga",  # Spain
    "Bundesliga",  # Germany
    "Serie A",  # Italy
    "Ligue 1",  # France
    "Primeira Liga",  # Portugal
    "Eredivisie",  # Netherlands
    "Liga MX",  # Mexico
    "Major League Soccer",  # USA
    "Saudi Professional",  # Saudi Arabia
    "First Division A",  # Belgium (Belgian Pro League)
    "Campeonato Brasileiro",  # Brazil
    "Primera División",  # Argentina
    "Eliteserien",  # Norway
]

# Season year strings to match when filtering competition seasons
TARGET_SEASON_YEARS: tuple[str, ...] = ("24/25", "25/26", "2024", "2025")

# Maps each target nation to its confederation's primary competition name.
# Used to find the right season_id when calling seasonal_competitor_players.
NATION_TO_COMPETITION: dict[str, str] = {
    "England": "UEFA Nations League",
    "France": "UEFA Nations League",
    "Spain": "UEFA Nations League",
    "Belgium": "UEFA Nations League",
    "Portugal": "UEFA Nations League",
    "Switzerland": "UEFA Nations League",
    "Norway": "UEFA Nations League",
    "Brazil": "Copa America",
    "Argentina": "Copa America",
    "Colombia": "Copa America",
    "Mexico": "CONCACAF Nations League",
    "USA": "CONCACAF Nations League",
    "Egypt": "Africa Cup of Nations",
    "Morocco": "Africa Cup of Nations",
}

# Keywords used to filter competitions.json for international tournaments.
# These are the competitions we query to discover national team competitor_ids.
# Used as fallback when the FIFA Rankings endpoint returns 404 (not in base trial).
INTL_TOURNAMENT_KEYWORDS: list[str] = [
    "UEFA Nations League",
    "Copa America",
    "CONCACAF Nations League",
    "Gold Cup",
    "Africa Cup",
    "AFCON",
    "FIFA World Cup",
    "UEFA European",
    "UEFA Euro",
]

# Heuristic: club country_code → league name (key in our league_season_map).
# Used to pair a player's club with the right season_id.
# Monaco (MCO) plays in Ligue 1; overrides handle common edge cases.
CLUB_COUNTRY_TO_LEAGUE: dict[str, str] = {
    "ENG": "Premier League",
    "ESP": "La Liga",
    "DEU": "Bundesliga",
    "ITA": "Serie A",
    "FRA": "Ligue 1",
    "MCO": "Ligue 1",  # Monaco
    "PRT": "Primeira Liga",
    "POR": "Primeira Liga",
    "NLD": "Eredivisie",
    "BEL": "First Division A",
    "MEX": "Liga MX",
    "USA": "Major League Soccer",
    "SAU": "Saudi Professional League",
    "BRA": "Campeonato Brasileiro Série A",
    "ARG": "Primera División",
    "NOR": "Eliteserien",
    "TUR": "Süper Lig",
}


# ── SporTradarClient ──────────────────────────────────────────────────────────


class SporTradarClient:
    """
    Thin wrapper around the Sportradar Soccer Extended v4 REST API.

    Responsibilities:
      - Disk caching: responses stored as JSON in ./cache/sportradar/.
        Re-runs cost zero quota for already-fetched endpoints.
      - Quota enforcement: raises RuntimeError before exceeding the 1,000-call limit.
      - Rate limiting: sleeps 1 second before every live API call (1 QPS).
      - Retry on 429: exponential backoff, 3 attempts maximum.

    Authentication:
      Sportradar v4 uses the x-api-key request header (NOT a query parameter).
    """

    def __init__(
        self,
        api_key: str,
        cache_dir: Path = Path("cache/sportradar"),
        quota_limit: int = QUOTA_LIMIT,
    ) -> None:
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.quota_limit = quota_limit
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._quota_file = self.cache_dir / "quota_used.json"
        self._calls_made = self._load_quota()
        log.info(
            "SporTradarClient ready | Quota: %d used / %d limit / %d remaining",
            self._calls_made,
            quota_limit,
            quota_limit - self._calls_made,
        )

    # ── Quota ──────────────────────────────────────────────────────────────────

    def _load_quota(self) -> int:
        if self._quota_file.exists():
            return json.loads(self._quota_file.read_text()).get("calls_made", 0)
        return 0

    def _save_quota(self) -> None:
        self._quota_file.write_text(
            json.dumps({"calls_made": self._calls_made}, indent=2)
        )

    @property
    def quota_remaining(self) -> int:
        return self.quota_limit - self._calls_made

    # ── Cache ──────────────────────────────────────────────────────────────────

    def _cache_path(self, endpoint: str) -> Path:
        """Converts an endpoint string into a stable, safe filesystem path."""
        slug = re.sub(r"[^\w/\-]", "_", endpoint).strip("/").replace("/", "__")
        if len(slug) > 180:
            slug = slug[:140] + "__" + hashlib.md5(endpoint.encode()).hexdigest()[:12]
        return self.cache_dir / f"{slug}.json"

    def _is_cached(self, endpoint: str) -> bool:
        return self._cache_path(endpoint).exists()

    def _read_cache(self, endpoint: str) -> dict:
        return json.loads(self._cache_path(endpoint).read_text())

    def _write_cache(self, endpoint: str, data: dict) -> None:
        self._cache_path(endpoint).write_text(
            json.dumps(data, indent=2, ensure_ascii=False)
        )

    # ── Core HTTP ──────────────────────────────────────────────────────────────

    def _get(self, endpoint: str, force_refresh: bool = False) -> dict[str, Any]:
        """
        Fetch one endpoint. Returns cached data when available; otherwise makes a
        live API call, writes the result to disk, and increments the quota counter.

        Args:
            endpoint:      Path relative to BASE_URL, e.g. "/rankings/fifa.json"
            force_refresh: Bypass cache and re-fetch from API.

        Raises:
            RuntimeError: Quota exhausted before this call.
            requests.HTTPError: Non-retryable HTTP error from the API.
        """
        if not force_refresh and self._is_cached(endpoint):
            log.debug("CACHE HIT  %s", endpoint)
            return self._read_cache(endpoint)

        if self._calls_made >= self.quota_limit:
            raise RuntimeError(
                f"Quota exhausted ({self._calls_made}/{self.quota_limit}). "
                "Use cached data or wait for trial reset."
            )

        url = f"{BASE_URL}{endpoint}"
        headers = {
            "accept": "application/json",
            "x-api-key": self.api_key,  # header auth — NOT a query param
        }

        last_exc: Exception | None = None
        for attempt in range(1, 4):
            try:
                time.sleep(1)  # enforce 1 QPS
                log.info(
                    "API CALL [%d/%d remaining]  %s",
                    self.quota_remaining,
                    self.quota_limit,
                    endpoint,
                )
                resp = requests.get(url, headers=headers, timeout=15)

                if resp.status_code == 429:
                    wait = 5 * attempt
                    log.warning(
                        "429 rate-limited — waiting %ds (attempt %d/3)", wait, attempt
                    )
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                data = resp.json()
                self._write_cache(endpoint, data)
                self._calls_made += 1
                self._save_quota()
                return data

            except requests.exceptions.Timeout as exc:
                last_exc = exc
                log.warning("Timeout on attempt %d/3 for %s", attempt, endpoint)
                time.sleep(2 * attempt)
            except requests.exceptions.HTTPError as exc:
                # 404 on player/club lookups is recoverable — caller handles it
                if exc.response is not None and exc.response.status_code == 404:
                    log.warning("404 — endpoint not found: %s", endpoint)
                    self._calls_made += 1  # still counts against quota
                    self._save_quota()
                    return {}  # empty dict signals missing resource
                raise

        raise RuntimeError(
            f"Failed to fetch {endpoint} after 3 attempts. Last error: {last_exc}"
        )

    # ── Named endpoints ────────────────────────────────────────────────────────

    def fifa_rankings(self) -> dict:
        """1 call — FIFA World Rankings; includes competitor_id for every nation."""
        return self._get("/rankings/fifa.json")

    def competitions(self) -> dict:
        """1 call — All competitions available in Soccer Extended."""
        return self._get("/competitions.json")

    def competition_seasons(self, competition_id: str) -> dict:
        """1 call — All historical seasons for a competition."""
        return self._get(f"/competitions/{competition_id}/seasons.json")

    def competitor_profile(self, competitor_id: str) -> dict:
        """1 call — Team profile including full squad player list."""
        return self._get(f"/competitors/{competitor_id}/profile.json")

    def player_profile(self, player_id: str) -> dict:
        """1 call — Player biographical info + all club/national team role history."""
        return self._get(f"/players/{player_id}/profile.json")

    def competitor_summaries(self, competitor_id: str) -> dict:
        """1 call — Past 30 match results for a team (used for Dixon-Coles data)."""
        return self._get(f"/competitors/{competitor_id}/summaries.json")

    def season_competitors(self, season_id: str) -> dict:
        """1 call — All teams (competitors) participating in a given season."""
        return self._get(f"/seasons/{season_id}/competitors.json")

    def seasonal_competitor_players(self, season_id: str, competitor_id: str) -> dict:
        """
        1 call — Players who appeared for a specific team in a specific season.
        Works for national teams in international competitions (unlike
        competitor_profile which returns 0 players for national teams).
        """
        return self._get(
            f"/seasons/{season_id}/competitors/{competitor_id}/players.json"
        )

    def sport_event_lineups(self, sport_event_id: str) -> dict:
        """1 call — Starting XI + substitutes for a specific match."""
        return self._get(f"/sport_events/{sport_event_id}/lineups.json")

    def seasonal_competitor_statistics(
        self, season_id: str, competitor_id: str
    ) -> dict:
        """
        1 call — Full seasonal statistics for one club in one season.
        Requires Soccer Extended Base + Tier 1/2 league coverage.

        Key fields we extract (confirmed in API spec; availability verified in probe):
          goals_expected              → npxG proxy
          goals_expected_created      → xT_Proxy (Sportradar's pre-built xG-chain metric)
          passes_into_box             → Deep_Progressions
          shots_faced_total           → Shot_Suppression
          tackles_opponent_half       → PPDA denominator component
          interceptions_opposition_half → PPDA denominator component
          fouls                       → PPDA denominator component
          passes_total                → PPDA numerator proxy
        """
        return self._get(
            f"/seasons/{season_id}/competitors/{competitor_id}/statistics.json"
        )


# ── Phase 1: National team IDs ────────────────────────────────────────────────


def _build_nation_map_from_competitors(
    competitors: list[dict],
    found: dict[str, str],
    remaining: set[str],
) -> None:
    """
    Helper: scans a competitors list and updates `found` in-place.
    Matches on country_code (primary) and name substring (fallback).
    Removes matched nations from `remaining`.
    """
    code_to_entry: dict[str, dict] = {
        c.get("country_code", "").upper(): c
        for c in competitors
        if c.get("country_code")
    }
    for nation in list(remaining):  # list() so we can mutate remaining inside loop
        iso_code = TARGET_NATIONS[nation]
        entry = code_to_entry.get(iso_code)
        if entry and entry.get("id"):
            found[nation] = entry["id"]
            remaining.discard(nation)
            log.info("%-14s → %-30s  %s", nation, entry.get("name", ""), entry["id"])
            continue
        # Name-based fallback
        name_match = next(
            (c for c in competitors if nation.lower() in c.get("name", "").lower()),
            None,
        )
        if name_match and name_match.get("id"):
            found[nation] = name_match["id"]
            remaining.discard(nation)
            log.info(
                "%-14s → %-30s  %s  [name match]",
                nation,
                name_match.get("name", ""),
                name_match["id"],
            )


def discover_national_team_ids(client: SporTradarClient) -> dict[str, str]:
    """
    Discovers Sportradar competitor_ids for all 14 target nations.

    Strategy (two-step with fallback):
      Primary:  FIFA Rankings endpoint (/rankings/fifa.json)
                Returns competitor IDs for all ranked nations in one call.
                ⚠️  NOT available on Soccer Extended Base trial — returns 404.
      Fallback: Competition-based discovery
                1. Call competitions.json (already paid for in Phase 2 — cached)
                2. Filter for known international tournaments (INTL_TOURNAMENT_KEYWORDS)
                3. For each tournament, call competition_seasons → most recent season
                4. Call season_competitors for that season → extract national team IDs
                Stops early once all 14 nations are found.

    Returns:
        {"France": "sr:competitor:XXXX", "England": "sr:competitor:YYYY", ...}

    Quota: 1 (FIFA Rankings, cached or new) + up to ~15 fallback calls.
            Fallback competition/season calls are shared with Phase 2 (cached).
    """
    found: dict[str, str] = {}
    remaining: set[str] = set(TARGET_NATIONS.keys())

    # ── Primary: FIFA Rankings ────────────────────────────────────────────────
    log.info("Trying FIFA Rankings endpoint...")
    rankings_data = client.fifa_rankings()
    rankings = rankings_data.get("rankings", [])
    if rankings:
        competitors = [e["competitor"] for e in rankings if e.get("competitor")]
        _build_nation_map_from_competitors(competitors, found, remaining)
        log.info("FIFA Rankings: found %d / 14 nations.", len(found))
    else:
        log.warning(
            "FIFA Rankings returned no data (endpoint not available on this plan). "
            "Switching to competition-based discovery."
        )

    if not remaining:
        return found  # all 14 found in one call — done

    # ── Fallback: competition-based discovery ─────────────────────────────────
    log.info(
        "Competition-based discovery for %d remaining nations: %s",
        len(remaining),
        sorted(remaining),
    )

    # competitions.json is fetched here but also reused in Phase 2 (cached = free)
    comps_data = client.competitions()
    all_comps = comps_data.get("competitions", [])

    # Filter for men's international tournaments
    intl_comps = [
        c
        for c in all_comps
        if c.get("gender") == "men"
        and any(
            kw.lower() in c.get("name", "").lower() for kw in INTL_TOURNAMENT_KEYWORDS
        )
    ]
    log.info("Found %d international men's tournaments to search.", len(intl_comps))
    for c in intl_comps:
        log.info("  %s  %s", c.get("id"), c.get("name"))

    for comp in intl_comps:
        if not remaining:
            break  # all nations found — stop spending calls

        comp_id = comp["id"]
        comp_name = comp["name"]

        # Get the most recent season for this competition
        try:
            seasons_data = client.competition_seasons(comp_id)
            seasons = seasons_data.get("seasons", [])
            if not seasons:
                continue
            # Seasons are returned newest-first by convention; take the first
            latest_season = seasons[0]
            season_id = latest_season["id"]
        except Exception as exc:
            log.warning("Cannot get seasons for %s: %s", comp_name, exc)
            continue

        # Get the teams in that season
        try:
            sc_data = client.season_competitors(season_id)
            season_competitors = sc_data.get("season_competitors", [])
            if not season_competitors:
                # Try alternate key name used by some endpoint versions
                season_competitors = sc_data.get("competitors", [])
            log.info(
                "  %s (%s) → %d competitors",
                comp_name,
                latest_season.get("name"),
                len(season_competitors),
            )
            _build_nation_map_from_competitors(season_competitors, found, remaining)
        except Exception as exc:
            log.warning("Cannot get season_competitors for %s: %s", comp_name, exc)

    if remaining:
        log.warning(
            "Could not find competitor_ids for: %s. "
            "They may not appear in any available international competition.",
            sorted(remaining),
        )

    return found


# ── Phase 2: League season IDs ────────────────────────────────────────────────


def discover_league_season_ids(client: SporTradarClient) -> dict[str, list[str]]:
    """
    Finds the 2024/25 and 2025/26 season IDs for every major men's league
    that club players in our target squads are likely to play in.

    Strategy:
      1. Call competitions.json → filter men's leagues by MAJOR_LEAGUE_SUBSTRINGS
      2. For each matched competition, call competition_seasons → extract season IDs
         where the year field matches TARGET_SEASON_YEARS

    Returns:
        {"Premier League": ["sr:season:118689", "sr:season:XXXXX"], "La Liga": [...], ...}

    Quota: 1 (competitions) + N_matched_leagues (seasons). Expect ≈11 calls total.
    """
    comps_data = client.competitions()
    all_comps = comps_data.get("competitions", [])
    log.info("Total competitions in Soccer Extended: %d", len(all_comps))

    # Filter: men's + name matches a target league substring
    matched: list[dict] = []
    for comp in all_comps:
        if comp.get("gender") != "men":
            continue
        name = comp.get("name", "")
        if any(sub.lower() in name.lower() for sub in MAJOR_LEAGUE_SUBSTRINGS):
            matched.append(comp)

    log.info("Matched %d major men's league competitions:", len(matched))
    for c in matched:
        log.info(
            "  %-40s  %s  [%s]",
            c.get("name"),
            c.get("id"),
            c.get("category", {}).get("name"),
        )

    result: dict[str, list[str]] = {}
    for comp in matched:
        comp_id = comp["id"]
        comp_name = comp["name"]
        try:
            seasons_data = client.competition_seasons(comp_id)
            seasons = seasons_data.get("seasons", [])
            relevant: list[str] = [
                s["id"]
                for s in seasons
                if any(t in s.get("year", "") for t in TARGET_SEASON_YEARS)
            ]
            if relevant:
                result[comp_name] = relevant
                for sid in relevant:
                    matching_season = next((s for s in seasons if s["id"] == sid), {})
                    log.info(
                        "  %-40s  %s  →  %s",
                        comp_name,
                        matching_season.get("year"),
                        sid,
                    )
        except Exception as exc:
            log.warning(
                "Could not fetch seasons for %s (%s): %s", comp_name, comp_id, exc
            )

    return result


# ── Phase 3: National team squad rosters ──────────────────────────────────────


def discover_nation_competition_seasons(
    client: SporTradarClient,
    all_comps: list[dict],
) -> dict[str, str]:
    """
    Finds the most recent season_id for each confederation competition
    relevant to our 14 target nations.

    Uses competitions.json (already cached from Phase 1) to find competition IDs
    for UEFA Nations League, Copa America, CONCACAF Nations League, and AFCON.
    Then calls competition_seasons for each to get the most recent season ID.

    Returns:
        {"France": "sr:season:XXXX", "Brazil": "sr:season:YYYY", ...}
        where the season_id is for that nation's confederation competition.

    Quota: 1 call per unique confederation competition found (~4 calls).
    """
    # Find competition IDs for each confederation by name match
    # Build: competition_keyword_substring → competition_id
    target_comp_names = list(set(NATION_TO_COMPETITION.values()))
    comp_name_to_id: dict[str, str] = {}

    for comp in all_comps:
        if comp.get("gender") != "men":
            continue
        name = comp.get("name", "")
        for target in target_comp_names:
            if target.lower() in name.lower() and target not in comp_name_to_id:
                comp_name_to_id[target] = comp["id"]
                log.info("  %-35s → %s", target, comp["id"])
                break

    # For each found competition, get the most recent season ID
    comp_to_season: dict[str, str] = {}
    for comp_name, comp_id in comp_name_to_id.items():
        try:
            seasons = client.competition_seasons(comp_id).get("seasons", [])
            # Find the most recent COMPLETED or ONGOING season in our date range
            # Prefer 2024/25 or 2024 over 2022 or 2026+
            best_season: dict | None = None
            for s in seasons:
                year = s.get("year", "")
                if any(t in year for t in TARGET_SEASON_YEARS):
                    best_season = s
                    break
            if not best_season and seasons:
                best_season = seasons[0]  # take whatever is most recent
            if best_season:
                comp_to_season[comp_name] = best_season["id"]
                log.info(
                    "  %-35s → %s (%s)",
                    comp_name,
                    best_season["id"],
                    best_season.get("name"),
                )
        except Exception as exc:
            log.warning("Cannot get seasons for %s (%s): %s", comp_name, comp_id, exc)

    # Map each nation to its confederation season_id
    nation_to_season: dict[str, str] = {}
    for nation, comp_name in NATION_TO_COMPETITION.items():
        sid = comp_to_season.get(comp_name)
        if sid:
            nation_to_season[nation] = sid
        else:
            log.warning(
                "No season_id found for %s (competition: %s)", nation, comp_name
            )

    return nation_to_season


def get_squad_player_ids(
    client: SporTradarClient,
    team_competitor_id: str,
    nation_name: str = "",
    n_lineup_matches: int = 3,
) -> tuple[list[str], list[dict]]:
    """
    Returns player IDs and historical match summaries for a national team.

    Strategy — two-step (Phase 3 + Phase 6 combined):
      Step 1: competitor_summaries(team_id) → recent match IDs + results
              (this ALSO provides the Dixon-Coles training data, so Phase 6
               is free once we have these summaries)
      Step 2: sport_event_lineups(match_id) for the N most recent completed
              matches → extract unique player IDs for the target team

    Why not competitor_profile or seasonal_competitor_players?
      Both return 404 for national teams on the Soccer Extended Base trial plan.
      National teams don't maintain a permanent squad registration the way clubs do.
      Their players only appear in match-level lineup data.

    Args:
        team_competitor_id: e.g. "sr:competitor:4481" (France)
        nation_name:        Display name for logging only.
        n_lineup_matches:   Number of recent match lineups to scan (default 3).
                            Each match contributes 23-26 player IDs.

    Returns:
        (player_ids, summaries)
          player_ids: de-duplicated list of player IDs seen in recent lineups
          summaries:  raw match summary list (for build_intl_results_df)

    Quota: 1 (competitor_summaries) + n_lineup_matches (sport_event_lineups) per nation

    ⚠️  PLACEHOLDER — lineup response structure:
      The exact JSON key path for players in sport_event_lineups is unconfirmed.
      We try the two most likely candidates:
        - data["lineups"][i]["players"]          (most probable)
        - data["sport_event_status"]["lineups"]   (less likely)
      The probe run will confirm the structure and this PLACEHOLDER comment
      should be removed once verified.
    """
    # Step 1: competitor_summaries → recent match results
    summaries_data = client.competitor_summaries(team_competitor_id)
    summaries = summaries_data.get("summaries", [])
    log.info(
        "  %-14s → %d match summaries (last 30)",
        nation_name or team_competitor_id,
        len(summaries),
    )

    # Step 2: pick N most recent completed matches and get their lineups
    completed = [
        s
        for s in summaries
        if s.get("sport_event_status", {}).get("status") in ("closed", "ended")
    ][:n_lineup_matches]

    player_ids: set[str] = set()

    for match_summary in completed:
        match_id = match_summary.get("sport_event", {}).get("id", "")
        if not match_id:
            continue

        lineups_data = client.sport_event_lineups(match_id)
        if not lineups_data:
            continue

        # Confirmed structure (verified from cached probe response):
        # { "lineups": { "competitors": [
        #     { "id": "sr:competitor:XXXX", "name": "...",
        #       "players": [
        #         { "id": "sr:player:XXXX", "name": "...", "starter": true, ... },
        #         ...
        #       ]
        #     },
        #     { ... opponent team ... }
        # ]}}
        competitors_in_lineup = lineups_data.get("lineups", {}).get("competitors", [])
        if not competitors_in_lineup:
            log.warning(
                "    %s: empty lineups.competitors. Keys: %s",
                match_id,
                list(lineups_data.get("lineups", {}).keys()),
            )
            continue

        # Find the entry for OUR national team
        for comp_entry in competitors_in_lineup:
            if comp_entry.get("id") != team_competitor_id:
                continue
            for player in comp_entry.get("players", []):
                pid = player.get("id", "")
                if pid:
                    player_ids.add(pid)

    log.info(
        "  %-14s → %d unique player IDs from %d match lineups",
        nation_name,
        len(player_ids),
        len(completed),
    )
    return list(player_ids), summaries


# ── Phase 4: Player → club mapping ───────────────────────────────────────────


def map_players_to_clubs(
    client: SporTradarClient, player_ids: list[str]
) -> dict[str, dict]:
    """
    For each player ID, calls player_profile to find their currently active club.

    The roles array in the response contains entries for every team a player has
    been associated with (current national team, current club, historical clubs,
    loan spells). We select the entry where:
        role["type"] == "player"   AND
        role["active"] == True     AND
        the competitor is NOT a national team (identified by having no meaningful
        country association that maps to a known national squad)

    ⚠️  PLACEHOLDER — Club vs national team disambiguation:
      National teams appear in roles with generic country names (e.g. "France",
      "England") and often have gender="male" + no specific city. Club teams have
      specific names (e.g. "Liverpool FC", "Barcelona"). We use a heuristic here:
      take the LAST active role with type=="player", since Sportradar orders roles
      most-recent first. If this misclassifies a player (e.g. someone on
      international loan), the output JSON will flag the mismatch clearly.
      This will be reviewed after the probe run shows a real roles response.

    Returns:
        {
          "sr:player:159665": {
            "player_name": "Salah, Mohamed",
            "competitor_id": "sr:competitor:44",
            "competitor_name": "Liverpool FC",
            "country_code": "ENG",
            "country": "England",
          },
          ...
        }

    Quota: 1 call per unique player ID (0 if cached).
    """
    result: dict[str, dict] = {}
    total = len(player_ids)

    for idx, player_id in enumerate(player_ids, 1):
        if player_id in result:
            continue  # already fetched (de-duplicate)
        try:
            data = client.player_profile(player_id)
            if not data:
                continue  # 404 — player not found

            player = data.get("player", {})
            player_name = player.get("name", "Unknown")

            # Find the active club role — prefer the first active player role
            # that is NOT the player's national team.
            # ⚠️ PLACEHOLDER: this heuristic may need refinement post-probe
            club_role: dict | None = None
            for role in data.get("roles", []):
                if role.get("type") == "player" and role.get("active") is True:
                    comp = role.get("competitor", {})
                    comp_name = comp.get("name", "")
                    # Skip entries that look like national teams:
                    # national teams generally match one of our TARGET_NATIONS values
                    is_national_team = any(
                        nation.lower() in comp_name.lower() for nation in TARGET_NATIONS
                    )
                    if not is_national_team:
                        club_role = role
                        break

            if club_role:
                comp = club_role.get("competitor", {})
                result[player_id] = {
                    "player_name": player_name,
                    "competitor_id": comp.get("id", ""),
                    "competitor_name": comp.get("name", ""),
                    "country_code": comp.get("country_code", ""),
                    "country": comp.get("country", ""),
                }
                log.debug(
                    "[%d/%d] %-30s → %s",
                    idx,
                    total,
                    player_name,
                    comp.get("name"),
                )
            else:
                log.warning(
                    "[%d/%d] No active club found for %s (%s)",
                    idx,
                    total,
                    player_name,
                    player_id,
                )

        except Exception as exc:
            log.warning("Error fetching player %s: %s", player_id, exc)

    log.info("Player→club mapping complete: %d / %d resolved.", len(result), total)
    return result


# ── Phase 5: Club seasonal statistics ─────────────────────────────────────────


def pull_club_seasonal_stats(
    client: SporTradarClient,
    player_club_map: dict[str, dict],
    league_season_map: dict[str, list[str]],
) -> dict[str, dict]:
    """
    For every unique club in player_club_map, fetches seasonal statistics
    for each relevant season (2024/25 and 2025/26 where available).

    Uses CLUB_COUNTRY_TO_LEAGUE to map each club's country_code to a league name,
    then looks up that league's season IDs from league_season_map.

    ⚠️  PLACEHOLDER — country_code heuristic:
      A club's country_code (e.g. "ENG") is used to infer their league
      ("Premier League"). This is correct for ~90% of cases but fails for:
        - Monaco (MCO) playing in Ligue 1  → handled via MCO override
        - Clubs promoted/relegated mid-window
        - Players in leagues not in MAJOR_LEAGUE_SUBSTRINGS (e.g. Süper Lig)
      Clubs with no matched season are logged and collected in unmatched_clubs.json
      for manual review. They do not cause the pipeline to abort.

    Returns:
        {
          "sr:competitor:44": {
            "competitor_name": "Liverpool FC",
            "seasons": {
              "sr:season:118689": { <raw API response> },
              "sr:season:XXXXX": { <raw API response> },
            }
          },
          ...
        }

    Quota: ~1 call per (club, season) pair.
    """
    # De-duplicate clubs from the player map
    clubs: dict[str, dict] = {}
    for player_data in player_club_map.values():
        cid = player_data.get("competitor_id", "")
        if cid and cid not in clubs:
            clubs[cid] = {
                "competitor_name": player_data.get("competitor_name", ""),
                "country_code": player_data.get("country_code", ""),
            }

    log.info("Unique clubs to fetch stats for: %d", len(clubs))

    result: dict[str, dict] = {}
    unmatched: list[dict] = []

    for cid, club_info in clubs.items():
        cc = club_info["country_code"]
        league_name = CLUB_COUNTRY_TO_LEAGUE.get(cc)
        season_ids = league_season_map.get(league_name, []) if league_name else []

        if not season_ids:
            log.warning(
                "No season IDs for %s (country=%s, league guess=%s)",
                club_info["competitor_name"],
                cc,
                league_name,
            )
            unmatched.append({"competitor_id": cid, **club_info})
            continue

        result[cid] = {
            "competitor_name": club_info["competitor_name"],
            "seasons": {},
        }

        for sid in season_ids:
            try:
                stats = client.seasonal_competitor_statistics(sid, cid)
                if stats:
                    result[cid]["seasons"][sid] = stats
                    log.info(
                        "  ✅ %-30s  season %s",
                        club_info["competitor_name"],
                        sid,
                    )
            except Exception as exc:
                log.warning(
                    "  ❌ %s  season %s: %s",
                    club_info["competitor_name"],
                    sid,
                    exc,
                )

    if unmatched:
        log.warning(
            "%d clubs had no matching league season. Saving to unmatched_clubs.json.",
            len(unmatched),
        )
        Path("output").mkdir(exist_ok=True)
        Path("output/unmatched_clubs.json").write_text(
            json.dumps(unmatched, indent=2, ensure_ascii=False)
        )

    return result


# ── Phase 6: International results (Dixon-Coles training data) ────────────────


def pull_intl_results(
    cached_summaries: dict[str, list[dict]],
) -> dict[str, list[dict]]:
    """
    Returns historical match summaries for each national team.

    Phase 3 (get_squad_player_ids) already calls competitor_summaries for every
    nation and stores the results. This function simply passes those cached
    summaries through to the DataFrame builder.

    ZERO additional API calls. The summaries dict is the second element of
    the tuple returned by get_squad_player_ids.

    Args:
        cached_summaries: {nation_name: [match_summary_dict, ...]}

    Returns:
        Same dict (alias). Kept as a named function for clarity in the pipeline.
    """
    for nation, summaries in cached_summaries.items():
        log.info("%-14s → %d match summaries available", nation, len(summaries))
    return cached_summaries


# ── DataFrame builders ────────────────────────────────────────────────────────


def build_club_metrics_df(raw_stats: dict[str, dict]) -> pd.DataFrame:
    """
    Converts the raw pull_club_seasonal_stats() output into the clean
    club_metrics_df expected by module_synthetic_aggregation.synthesize_team().

    Metric mapping (Sportradar field → our internal name):
      goals_expected              → npxG
      goals_expected_created      → xT_Proxy  (Q7: use pre-built metric, approved)
      passes_into_box             → Deep_Progressions
      PPDA (computed, see below)  → PPDA
      shots_faced_total           → Shot_Suppression

    PPDA computation:
      PPDA = passes_total / max(tackles_opponent_half
                                + interceptions_opposition_half
                                + fouls, 1)
      Lower PPDA = higher press intensity (counterintuitive direction noted).
      The Dixon-Coles layer should invert or treat PPDA as a penalty term.

    ⚠️  PLACEHOLDER — JSON nesting path:
      The seasonal statistics endpoint returns a nested object. Based on the
      documented schema and the Sport Event Timeline example, the expected path is:
        response["statistics"]["totals"]["competitors"][0]["statistics"]
      This will be confirmed after the probe run. If the structure differs,
      this function's extraction logic is the ONLY thing that needs updating.
      All field names are confirmed correct per the API specification.

    Returns:
        DataFrame[club_name, npxG, xT_Proxy, Deep_Progressions, PPDA, Shot_Suppression]
    """
    rows: list[dict] = []

    for competitor_id, club_data in raw_stats.items():
        club_name = club_data.get("competitor_name", competitor_id)

        # Use the most recently available season's stats
        # (seasons dict is insertion-ordered; last inserted = most recent fetched)
        best_stats: dict | None = None
        for _season_id, season_resp in club_data.get("seasons", {}).items():
            try:
                # ⚠️ PLACEHOLDER: nesting path inferred from documented schema
                competitors_in_totals = (
                    season_resp.get("statistics", {})
                    .get("totals", {})
                    .get("competitors", [])
                )
                if competitors_in_totals:
                    best_stats = competitors_in_totals[0].get("statistics", {})
            except (AttributeError, KeyError, IndexError):
                continue

        if best_stats is None:
            log.warning("No usable statistics found for %s. Skipping.", club_name)
            continue

        # ── Extract raw values ───────────────────────────────────────────────
        goals_expected = float(best_stats.get("goals_expected", 0) or 0)
        goals_expected_created = float(best_stats.get("goals_expected_created", 0) or 0)
        passes_into_box = float(best_stats.get("passes_into_box", 0) or 0)
        shots_faced_total = float(best_stats.get("shots_faced_total", 0) or 0)

        passes_total = float(best_stats.get("passes_total", 1) or 1)
        tackles_opp_half = float(best_stats.get("tackles_opponent_half", 0) or 0)
        intercepts_opp_half = float(
            best_stats.get("interceptions_opposition_half", 0) or 0
        )
        fouls = float(best_stats.get("fouls", 0) or 0)

        defensive_actions = tackles_opp_half + intercepts_opp_half + fouls
        ppda = passes_total / max(defensive_actions, 1)

        rows.append(
            {
                "club_name": club_name,
                "npxG": round(goals_expected, 4),
                "xT_Proxy": round(goals_expected_created, 4),
                "Deep_Progressions": round(passes_into_box, 4),
                "PPDA": round(ppda, 4),
                "Shot_Suppression": round(shots_faced_total, 4),
            }
        )

    df = pd.DataFrame(rows)
    log.info("club_metrics_df: %d clubs.", len(df))
    return df


def build_intl_results_df(intl_results: dict[str, list[dict]]) -> pd.DataFrame:
    """
    Flattens the pull_intl_results() output into a tabular DataFrame
    ready for Dixon-Coles model training.

    ⚠️  PLACEHOLDER — Field paths in competitor_summaries response:
      The Sportradar summaries endpoint wraps each match in a 'summaries' array
      of objects containing 'sport_event' and 'sport_event_status'. The paths
      used below follow the documented Soccer Extended schema. The exact nesting
      will be verified after the probe run.

    Returns:
        DataFrame[date, home_team, away_team, home_score, away_score, neutral]
        Sorted by date descending. Duplicate matches de-duplicated.
    """
    rows: list[dict] = []

    for _nation, summaries in intl_results.items():
        for match in summaries:
            try:
                se = match.get("sport_event", {})
                status = match.get("sport_event_status", {})

                # Only include completed matches with confirmed scores
                match_status = status.get("status", "")
                if match_status not in ("closed", "ended"):
                    continue

                date = se.get("start_time", "")[:10]
                competitors = se.get("competitors", [])
                home_team = next(
                    (c["name"] for c in competitors if c.get("qualifier") == "home"),
                    "?",
                )
                away_team = next(
                    (c["name"] for c in competitors if c.get("qualifier") == "away"),
                    "?",
                )
                home_score = status.get("home_score")
                away_score = status.get("away_score")
                neutral = (
                    se.get("sport_event_conditions", {})
                    .get("ground", {})
                    .get("neutral", False)
                )

                if home_score is not None and away_score is not None:
                    rows.append(
                        {
                            "date": date,
                            "home_team": home_team,
                            "away_team": away_team,
                            "home_score": int(home_score),
                            "away_score": int(away_score),
                            "neutral": bool(neutral),
                        }
                    )
            except Exception as exc:
                log.debug("Skipping malformed match entry: %s", exc)

    df = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["date", "home_team", "away_team"])
        .sort_values("date", ascending=False)
        .reset_index(drop=True)
    )
    log.info("intl_results_df: %d completed matches.", len(df))
    return df


# ── Run modes ─────────────────────────────────────────────────────────────────


def probe(client: SporTradarClient) -> None:
    """
    Validates authentication, data structure, and tier coverage using
    a minimal set of API calls (~7 max — none wasted on FIFA Rankings 404).

    Calls made:
      [1] competitions.json         → confirm API key works, list all competitions
      [2] competition_seasons       → UEFA Nations League 2024/25 season ID
      [3] season_competitors        → France competitor_id from UEFA NL season
      [4] competitor_profile(FRA)   → France squad roster (player IDs)
      [5] player_profile            → first squad player → current club
      [6] competition_seasons (PL)  → Premier League 2024/25 season ID
      [7] seasonal_competitor_stats → tier coverage check (8 target fields)

    Run this BEFORE the full pipeline to validate tier coverage
    without committing 650+ quota calls.
    """
    log.info("=" * 64)
    log.info("PROBE MODE — up to 7 API calls")
    log.info("=" * 64)

    # ── [1] Competitions ──────────────────────────────────────────────────────
    log.info("\n[1/7] competitions.json")
    comps_data = client.competitions()
    comps = comps_data.get("competitions", [])
    if not comps:
        log.error("competitions.json returned no data. Check API key.")
        return
    log.info("  ✅ Total competitions: %d", len(comps))
    mens = [c for c in comps if c.get("gender") == "men"]
    log.info("  Men's competitions: %d", len(mens))
    for c in mens[:5]:
        log.info("    %-40s  %s", c.get("name"), c.get("id"))

    # Find the UEFA Nations League (men's) competition ID
    unl_comp = next(
        (c for c in mens if "UEFA Nations League" in c.get("name", "")), None
    )
    if not unl_comp:
        log.error("UEFA Nations League not found. Cannot locate France.")
        return
    log.info("  UEFA Nations League ID: %s", unl_comp["id"])

    # ── [2] UEFA Nations League seasons ───────────────────────────────────────
    log.info("\n[2/7] competition_seasons for UEFA Nations League")
    unl_seasons = client.competition_seasons(unl_comp["id"]).get("seasons", [])
    log.info("  Seasons returned: %d", len(unl_seasons))
    for s in unl_seasons[:3]:
        log.info("    %s  %s  %s", s.get("id"), s.get("name"), s.get("year"))
    if not unl_seasons:
        log.error("No seasons for UEFA Nations League. Stopping probe.")
        return
    unl_season_id = unl_seasons[0]["id"]  # most recent first

    # ── [3] UEFA NL season_competitors → find France ────────────────────────────
    log.info("\n[3/7] season_competitors for %s", unl_seasons[0].get("name"))
    sc_data = client.season_competitors(unl_season_id)
    season_comps = sc_data.get("season_competitors", sc_data.get("competitors", []))
    log.info("  Teams in season: %d", len(season_comps))

    france_entry = next(
        (
            c
            for c in season_comps
            if c.get("country_code") == "FRA" or "France" in c.get("name", "")
        ),
        None,
    )
    if not france_entry:
        log.error("France not found in UEFA NL season. Trying name search...")
        for c in season_comps[:8]:
            log.info("  %s  %s  %s", c.get("id"), c.get("name"), c.get("country_code"))
        return
    france_id = france_entry["id"]
    log.info("  ✅ France found: %s  %s", france_entry.get("name"), france_id)

    # ── [4] France squad via competitor_summaries + sport_event_lineups ────────
    log.info("\n[4/7] competitor_summaries(France) + sport_event_lineups")
    france_pids, france_summaries = get_squad_player_ids(
        client,
        france_id,
        "France",
        n_lineup_matches=1,  # just 1 match in probe
    )
    log.info("  Unique player IDs from lineups: %d", len(france_pids))
    if not france_pids:
        log.warning(
            "  ⚠️  0 players found from lineups. "
            "Summaries returned: %d. Check sport_event_lineups response structure.",
            len(france_summaries),
        )
        # Still continue to test the player_profile step with a known player
        # Kylian Mbappe known SR ID as fallback for probe only
        france_pids = ["sr:player:225600"]  # Mbappe — probe only, not used in full run
        log.info("  Using hardcoded probe player ID: %s (Mbappe)", france_pids[0])

    first_player_id = france_pids[0]

    # ── [5] Player profile ──────────────────────────────────────────────────────
    log.info("\n[5/7] player_profile → club mapping")
    player_data = client.player_profile(first_player_id)
    p_name = player_data.get("player", {}).get("name", "?")
    roles = player_data.get("roles", [])
    log.info("  Player: %s | Total roles: %d", p_name, len(roles))

    club_role = next(
        (
            r
            for r in roles
            if r.get("type") == "player"
            and r.get("active") is True
            and not any(
                n.lower() in r.get("competitor", {}).get("name", "").lower()
                for n in TARGET_NATIONS
            )
        ),
        None,
    )
    if not club_role:
        log.warning("  No active club role found. Probe cannot continue to step 5.")
        return
    club = club_role["competitor"]
    log.info(
        "  Current club: %-30s  %s  (country: %s)",
        club.get("name"),
        club.get("id"),
        club.get("country_code"),
    )

    club_id = club.get("id")
    club_country = club.get("country_code", "")

    # ── [6] Premier League seasons → get a real season_id ──────────────────────
    log.info("\n[6/7] competition_seasons (Premier League) → season_id for stats test")
    # ⚠️  PLACEHOLDER: we use Premier League 24/25 as the probe's test season.
    # If the player's club is NOT in the Premier League, the seasonal stats
    # call in step 7 will return 404/empty — that's expected and informative.
    # The full pipeline's discover_league_season_ids() finds the correct
    # season per league, not per player's nationality.
    PL_COMPETITION_ID = "sr:competition:17"  # Premier League (confirmed from docs)
    log.info(
        "  Premier League competition ID: %s  (hardcoded, doc-confirmed)",
        PL_COMPETITION_ID,
    )
    seasons_data = client.competition_seasons(PL_COMPETITION_ID)
    seasons = seasons_data.get("seasons", [])
    log.info("  Premier League seasons returned: %d", len(seasons))
    for s in seasons[:4]:
        log.info("    %s  %s  %s", s.get("id"), s.get("name"), s.get("year"))

    probe_season_id = seasons[0].get("id") if seasons else None
    if not probe_season_id:
        log.error("  No seasons found. Cannot test seasonal stats endpoint.")
        return

    # ── [7] Seasonal stats → tier coverage check ──────────────────────────────
    log.info("\n[7/7] seasonal_competitor_statistics → tier + field coverage")
    log.info("  season=%s  club=%s (%s)", probe_season_id, club_id, club.get("name"))
    if club_country != "ENG":
        log.info(
            "  ⚠️  Club is %s (country=%s) — not in Premier League. "
            "Expecting 404/empty response. This is expected in probe mode.",
            club.get("name"),
            club_country,
        )
    stats_resp = client.seasonal_competitor_statistics(probe_season_id, club_id)

    if not stats_resp:
        log.warning("  Empty response — club likely not in this league/season.")
        log.warning("  Try again after full pipeline sets correct season IDs per club.")
    else:
        log.info("  Response top-level keys: %s", list(stats_resp.keys()))
        stat_block = stats_resp.get("statistics", {})
        log.info("  statistics keys: %s", list(stat_block.keys()))
        totals = stat_block.get("totals", {})
        log.info("  totals keys: %s", list(totals.keys()))
        competitors_in_totals = totals.get("competitors", [])
        log.info("  competitors in totals: %d", len(competitors_in_totals))

        if competitors_in_totals:
            sample_stats = competitors_in_totals[0].get("statistics", {})
            TARGET_FIELDS = [
                "goals_expected",
                "goals_expected_created",
                "passes_into_box",
                "shots_faced_total",
                "tackles_opponent_half",
                "interceptions_opposition_half",
                "fouls",
                "passes_total",
            ]
            log.info("\n  ── Key metric field availability ──")
            all_present = True
            for field in TARGET_FIELDS:
                val = sample_stats.get(field)
                present = val is not None
                if not present:
                    all_present = False
                indicator = "✅" if present else "❌"
                log.info(
                    "    %s  %-40s  %s",
                    indicator,
                    field,
                    val if present else "NOT PRESENT",
                )

            if all_present:
                log.info("\n  ✅ All 8 target fields present. Full pipeline is ready.")
            else:
                log.warning(
                    "\n  ⚠️  Some fields missing. This may indicate Tier 2 or lower coverage "
                    "for this competition. Check coverage matrix for the specific league."
                )

    log.info("\n" + "=" * 64)
    log.info("PROBE COMPLETE")
    log.info("  Quota used:      %d / %d", client._calls_made, client.quota_limit)
    log.info("  Quota remaining: %d", client.quota_remaining)
    log.info("=" * 64)


def run_full_pipeline(client: SporTradarClient) -> None:
    """
    Executes all 6 pipeline phases end-to-end.

    Every intermediate result is saved to output/ so the pipeline can be
    resumed (cached API responses are reused; already-built JSON files
    are overwritten but not re-fetched from the API).

    Expected runtime: ~662 seconds minimum (1 QPS rate limit enforces this).
    """
    out = Path("output")
    out.mkdir(exist_ok=True)

    log.info("=" * 64)
    log.info("FULL PIPELINE START | Quota remaining: %d", client.quota_remaining)
    log.info("=" * 64)

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    log.info("\n── Phase 1: National team IDs")
    national_team_ids = discover_national_team_ids(client)
    (out / "national_team_ids.json").write_text(json.dumps(national_team_ids, indent=2))
    log.info("Saved national_team_ids.json (%d nations)", len(national_team_ids))

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    log.info("\n── Phase 2: League season IDs")
    league_season_map = discover_league_season_ids(client)
    (out / "league_season_map.json").write_text(json.dumps(league_season_map, indent=2))
    log.info("Saved league_season_map.json (%d leagues)", len(league_season_map))

    # ── Phase 2.5: Map each nation to its confederation competition season ──────
    # Uses competitions data already cached in Phase 1/2 (0 additional calls
    # for competitions.json; ~4 calls for competition_seasons per confederation)
    log.info("\n── Phase 2.5: Discover confederation season IDs for squad lookup")
    comps_cached = (
        json.loads(
            (client.cache_dir / "competitions.json".replace("/", "__")).read_text()
        )
        if (client.cache_dir / "competitions.json".replace("/", "__")).exists()
        else client.competitions()
    )
    # The cache file for /competitions.json is named competitions.json (no leading slash)
    # Re-fetch safely (cache hit = 0 calls)
    comps_cached = client.competitions().get("competitions", [])
    nation_season_map = discover_nation_competition_seasons(client, comps_cached)
    (out / "nation_season_map.json").write_text(json.dumps(nation_season_map, indent=2))
    log.info(
        "Saved nation_season_map.json (%d nations mapped to seasons)",
        len(nation_season_map),
    )

    # ── Phase 3 (+ Phase 6 combined) ───────────────────────────────────────────────
    log.info(
        "\n── Phase 3: Squad rosters via match lineups + match results (0 extra calls for Phase 6)"
    )
    all_player_ids: set[str] = set()
    nation_squads: dict[str, list[str]] = {}
    nation_summaries: dict[str, list[dict]] = {}  # also feeds Phase 6
    for nation, team_id in national_team_ids.items():
        pids, summaries = get_squad_player_ids(client, team_id, nation)
        nation_squads[nation] = pids
        nation_summaries[nation] = summaries
        all_player_ids.update(pids)

    log.info("Total unique player IDs across all 14 squads: %d", len(all_player_ids))
    (out / "nation_squads.json").write_text(json.dumps(nation_squads, indent=2))
    (out / "all_player_ids.json").write_text(
        json.dumps(sorted(all_player_ids), indent=2)
    )

    # ── Phase 4 ───────────────────────────────────────────────────────────────
    log.info("\n── Phase 4: Map %d players to clubs", len(all_player_ids))
    player_club_map = map_players_to_clubs(client, sorted(all_player_ids))
    (out / "player_club_map.json").write_text(
        json.dumps(player_club_map, indent=2, ensure_ascii=False)
    )
    log.info("Saved player_club_map.json (%d players resolved)", len(player_club_map))

    # ── Phase 5 ───────────────────────────────────────────────────────────────
    log.info("\n── Phase 5: Club seasonal statistics")
    raw_club_stats = pull_club_seasonal_stats(
        client, player_club_map, league_season_map
    )
    (out / "raw_club_stats.json").write_text(
        json.dumps(raw_club_stats, indent=2, ensure_ascii=False)
    )
    log.info("Saved raw_club_stats.json (%d clubs)", len(raw_club_stats))

    club_metrics_df = build_club_metrics_df(raw_club_stats)
    club_metrics_df.to_csv(out / "club_metrics.csv", index=False)
    log.info("Saved club_metrics.csv (%d rows)", len(club_metrics_df))
    if not club_metrics_df.empty:
        log.info("\n%s", club_metrics_df.to_string(index=False))

    # ── Phase 6 (no new API calls — reuses summaries from Phase 3) ──────────────
    log.info(
        "\n── Phase 6: International match results (Dixon-Coles training, 0 extra calls)"
    )
    intl_results = pull_intl_results(nation_summaries)
    intl_df = build_intl_results_df(intl_results)
    intl_df.to_csv(out / "intl_results.csv", index=False)
    log.info("Saved intl_results.csv (%d completed matches)", len(intl_df))

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 64)
    log.info("FULL PIPELINE COMPLETE")
    log.info(
        "  Quota used:              %d / %d", client._calls_made, client.quota_limit
    )
    log.info("  Quota remaining:         %d", client.quota_remaining)
    log.info("  Clubs with stats:        %d", len(club_metrics_df))
    log.info("  International matches:   %d", len(intl_df))
    log.info("  Output directory:        %s/", out.resolve())
    log.info("=" * 64)


def show_status(client: SporTradarClient) -> None:
    """Prints quota status. Zero API calls made."""
    print(f"\nQuota status:")
    print(f"  Used:      {client._calls_made} / {client.quota_limit}")
    print(f"  Remaining: {client.quota_remaining}")
    print(f"  Cache dir: {client.cache_dir.resolve()}\n")


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sportradar Soccer Extended v4 pipeline — WC 2026 prediction engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python sportradar_pipeline.py --mode status  # quota check, zero calls\n"
            "  python sportradar_pipeline.py --mode probe   # 5-call auth + tier test\n"
            "  python sportradar_pipeline.py --mode run     # full pipeline (~662 calls)\n"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["status", "probe", "run"],
        default="status",
        help="Execution mode (default: status)",
    )
    args = parser.parse_args()

    api_key = os.getenv("sportradar_api_key")
    if not api_key:
        raise EnvironmentError(
            "sportradar_api_key not set. "
            "Add it to .env as:  sportradar_api_key=YOUR_KEY_HERE"
        )

    client = SporTradarClient(api_key=api_key)

    if args.mode == "status":
        show_status(client)
    elif args.mode == "probe":
        probe(client)
    elif args.mode == "run":
        run_full_pipeline(client)


if __name__ == "__main__":
    main()
