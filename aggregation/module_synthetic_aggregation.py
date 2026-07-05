"""
module_synthetic_aggregation.py
================================
1/11 Minute-Weighted Club Aggregation Engine
---------------------------------------------
Synthesizes a national team's tactical profile by aggregating the domestic
club metrics of the 11 players in the starting lineup. Each player's
contribution is weighted by four factors:

    Player_Contribution = Club_Metric * (
        Positional_Weight * League_Coefficient * Minutes_Played_Pct * Usage_Rate_Pct
    )

The 11 player contributions are then summed to produce a single synthetic
metric row representing the national team.

Note on PPDA directionality
----------------------------
PPDA (Passes Per Defensive Action) is an inverse metric — a lower value
means MORE intense pressing. This module computes it using the same additive
formula as all other metrics so it remains scale-consistent when fed into the
Dixon-Coles feature matrix. The caller is responsible for inverting or
normalising PPDA before fitting the model.

Author: Vibepredicting-Football Engine
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SyntheticAggregation")


# ─────────────────────────────────────────────────────────────────────────────
# 1. LEAGUE STRENGTH COEFFICIENTS
#    Reflects the quality/competitiveness of each domestic league.
#    Premier League is the baseline at 1.0. All others are discounted
#    relative to it based on UEFA coefficients and transfer market data.
# ─────────────────────────────────────────────────────────────────────────────
LEAGUE_COEFFICIENTS: dict[str, float] = {
    # Tier 1 — Elite
    "Premier League": 1.00,
    "La Liga": 0.97,
    "Bundesliga": 0.95,
    "Serie A": 0.93,
    "Ligue 1": 0.90,
    # Tier 2 — Strong
    "Primeira Liga": 0.82,
    "Eredivisie": 0.80,
    "Championship": 0.78,
    "Liga MX": 0.75,
    "Brazilian Serie A": 0.74,
    "Argentine Primera": 0.73,
    "Scottish Premiership": 0.72,
    "Belgian Pro League": 0.71,
    "Süper Lig": 0.70,
    # Tier 3 — Moderate
    "MLS": 0.65,
    "Saudi Pro League": 0.60,
    "Chinese Super League": 0.55,
    # Tier 4 — Developing
    "J1 League": 0.50,
    "K League 1": 0.48,
}

# Default fallback for leagues not listed above
_DEFAULT_LEAGUE_COEFFICIENT: float = 0.60


# ─────────────────────────────────────────────────────────────────────────────
# 2. POSITIONAL WEIGHTS
#    A nested dict: metric → role → weight.
#    Constraint: for every metric, sum of all 11 canonical role weights = 1.0.
#
#    The 11 canonical roles are:
#      Goalkeeper, Right Back, Left Back, Center Back,
#      Defensive Midfield, Central Midfield,
#      Right Wing, Left Wing, Attacking Midfield,
#      Second Striker, Striker
#
#    Design rationale per metric:
#      PPDA            → pressing load falls on midfielders and forwards
#      Shot_Suppression → defensive solidity is the back-line & GK's job
# ─────────────────────────────────────────────────────────────────────────────
POSITIONAL_WEIGHTS: dict[str, dict[str, float]] = {
    # ── PPDA: Passes Per Defensive Action (lower = more intense pressing) ────
    # High-press systems are driven by the front three and central mids.
    # GK does not participate in the pressing trap.
    "PPDA": {
        "Goalkeeper": 0.00,
        "Right Back": 0.05,
        "Left Back": 0.05,
        "Center Back": 0.05,
        "Defensive Midfield": 0.15,
        "Central Midfield": 0.15,
        "Right Wing": 0.12,
        "Left Wing": 0.12,
        "Attacking Midfield": 0.13,
        "Second Striker": 0.09,
        "Striker": 0.09,
        # sum = 1.00
    },
    # ── Shot_Suppression: Open-play shots allowed per 90 ────────────────────
    # Shot prevention is overwhelmingly a back-line responsibility.
    # GK earns the highest single-role weight as the last line of defence.
    "Shot_Suppression": {
        "Goalkeeper": 0.20,
        "Right Back": 0.13,
        "Left Back": 0.13,
        "Center Back": 0.20,
        "Defensive Midfield": 0.12,
        "Central Midfield": 0.08,
        "Right Wing": 0.04,
        "Left Wing": 0.04,
        "Attacking Midfield": 0.03,
        "Second Striker": 0.02,
        "Striker": 0.01,
        # sum = 1.00
    },
}

# Canonical metrics this module operates on (must match club_metrics_df columns)
_METRICS: list[str] = list(POSITIONAL_WEIGHTS.keys())

# ─────────────────────────────────────────────────────────────────────────────
# 3. ROLE ALIASES
#    Maps common free-text role strings to a canonical role key.
#    Allows the roster JSON to use shorthand or alternate naming.
# ─────────────────────────────────────────────────────────────────────────────
ROLE_ALIASES: dict[str, str] = {
    # Goalkeeper
    "gk": "Goalkeeper",
    "keeper": "Goalkeeper",
    "goalie": "Goalkeeper",
    # Defenders
    "cb": "Center Back",
    "centre back": "Center Back",
    "centreback": "Center Back",
    "rb": "Right Back",
    "right back": "Right Back",
    "rightback": "Right Back",
    "lb": "Left Back",
    "left back": "Left Back",
    "leftback": "Left Back",
    "wb": "Right Back",  # Wide-back defaults to Right Back; caller should be explicit
    "rwb": "Right Back",
    "lwb": "Left Back",
    # Midfielders
    "dm": "Defensive Midfield",
    "cdm": "Defensive Midfield",
    "defensive mid": "Defensive Midfield",
    "defensive midfielder": "Defensive Midfield",
    "cm": "Central Midfield",
    "central mid": "Central Midfield",
    "central midfielder": "Central Midfield",
    "box-to-box": "Central Midfield",
    "am": "Attacking Midfield",
    "cam": "Attacking Midfield",
    "attacking mid": "Attacking Midfield",
    "attacking midfielder": "Attacking Midfield",
    "trequartista": "Attacking Midfield",
    # Wingers
    "rw": "Right Wing",
    "right wing": "Right Wing",
    "right winger": "Right Wing",
    "lw": "Left Wing",
    "left wing": "Left Wing",
    "left winger": "Left Wing",
    "winger": "Right Wing",
    # Forwards
    "ss": "Second Striker",
    "second striker": "Second Striker",
    "shadow striker": "Second Striker",
    "cf": "Striker",
    "st": "Striker",
    "striker": "Striker",
    "centre forward": "Striker",
    "center forward": "Striker",
    "false 9": "Attacking Midfield",
}


# ─────────────────────────────────────────────────────────────────────────────
# 4. HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_role(raw_role: str) -> str:
    """
    Normalise a free-text role string to a canonical POSITIONAL_WEIGHTS key.
    Matching is case-insensitive. Raises ValueError for unrecognised roles.
    """
    key = raw_role.strip().lower()
    # Direct canonical match (case-insensitive)
    canonical_roles_lower = {
        r.lower(): r for r in next(iter(POSITIONAL_WEIGHTS.values()))
    }
    if key in canonical_roles_lower:
        return canonical_roles_lower[key]
    # Alias lookup
    if key in ROLE_ALIASES:
        return ROLE_ALIASES[key]
    raise ValueError(
        f"Unrecognised role '{raw_role}'. "
        f"Add it to ROLE_ALIASES or use one of: {list(canonical_roles_lower.values())}"
    )


def _resolve_league_coefficient(league: str) -> float:
    """Return the league strength coefficient, falling back to the default."""
    coeff = LEAGUE_COEFFICIENTS.get(league)
    if coeff is None:
        logger.warning(
            "League '%s' not found in LEAGUE_COEFFICIENTS. "
            "Using default coefficient %.2f.",
            league,
            _DEFAULT_LEAGUE_COEFFICIENT,
        )
        return _DEFAULT_LEAGUE_COEFFICIENT
    return coeff


def _validate_positional_weights() -> None:
    """
    Integrity check: assert each metric's positional weights sum to 1.0.
    Called once at module load. Raises AssertionError if misconfigured.
    """
    for metric, weights in POSITIONAL_WEIGHTS.items():
        total = round(sum(weights.values()), 10)
        assert abs(total - 1.0) < 1e-6, (
            f"Positional weights for '{metric}' sum to {total:.6f}, expected 1.0. "
            f"Fix the POSITIONAL_WEIGHTS dictionary."
        )
    logger.debug("Positional weight integrity check passed for all metrics.")


# Run validation at import time — fail fast if the config is broken.
_validate_positional_weights()


# ─────────────────────────────────────────────────────────────────────────────
# 5. CORE AGGREGATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────


def synthesize_team(
    roster_json: list[dict[str, Any]],
    club_metrics_df: pd.DataFrame,
    team_name: str = "Synthetic XI",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Synthesise a national team's tactical profile from 11 players' club data.

    Parameters
    ----------
    roster_json : list[dict]
        List of exactly 11 player dicts. Each dict must contain:
            - player_name       (str)
            - club              (str)   must match `club_name` in club_metrics_df
            - league            (str)   used to look up LEAGUE_COEFFICIENTS
            - role              (str)   player's primary role (canonical or aliased)
            - minutes_played_pct (float) 0.0–1.0 share of possible minutes played
            - usage_rate_pct    (float) 0.0–1.0 proportion of team actions involving player

    club_metrics_df : pd.DataFrame
        Must contain column `club_name` plus all five metric columns:
        npxG, xT_Proxy, Deep_Progressions, PPDA, Shot_Suppression.
        Values should be per-90 normalised.

    team_name : str
        Label for the synthesised team row (stored in the `team` column).

    verbose : bool
        If True, logs each player's computed contributions for diagnostics.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with columns:
        [team, npxG, xT_Proxy, Deep_Progressions, PPDA, Shot_Suppression]
        representing the summed synthetic profile of the national team.

    Raises
    ------
    ValueError
        If roster_json does not contain exactly 11 players, or required fields
        are missing, or a club is not found in club_metrics_df.
    """
    # ── Validation ───────────────────────────────────────────────────────────
    if len(roster_json) != 11:
        raise ValueError(
            f"roster_json must contain exactly 11 players, got {len(roster_json)}."
        )

    required_player_fields = {
        "player_name",
        "club",
        "league",
        "role",
        "minutes_played_pct",
        "usage_rate_pct",
    }
    for i, player in enumerate(roster_json):
        missing = required_player_fields - set(player.keys())
        if missing:
            raise ValueError(
                f"Player at index {i} is missing required fields: {missing}"
            )

    required_df_cols = {"club_name"} | set(_METRICS)
    missing_cols = required_df_cols - set(club_metrics_df.columns)
    if missing_cols:
        raise ValueError(f"club_metrics_df is missing columns: {missing_cols}")

    # Build a lookup dict for O(1) club access: {club_name: {metric: value}}
    club_lookup: dict[str, dict[str, float]] = club_metrics_df.set_index("club_name")[
        _METRICS
    ].to_dict(orient="index")

    # ── Accumulate contributions ─────────────────────────────────────────────
    # Initialise accumulator to zero for each metric
    team_totals: dict[str, float] = {m: 0.0 for m in _METRICS}

    if verbose:
        logger.info("=" * 62)
        logger.info("Synthesising team: %s", team_name)
        logger.info("=" * 62)

    for player in roster_json:
        p_name = player["player_name"]
        p_club = player["club"]
        p_league = player["league"]
        p_role_raw = player["role"]
        p_minutes_pct = float(player["minutes_played_pct"])
        p_usage_pct = float(player["usage_rate_pct"])

        # Resolve role → canonical key
        p_role = _resolve_role(p_role_raw)

        # Look up league coefficient
        league_coeff = _resolve_league_coefficient(p_league)

        # Look up club metrics
        if p_club not in club_lookup:
            raise ValueError(
                f"Club '{p_club}' (player: {p_name}) not found in club_metrics_df. "
                f"Available clubs: {list(club_lookup.keys())}"
            )
        club_metrics = club_lookup[p_club]

        if verbose:
            logger.info(
                "  %-22s | %-20s | %-18s | league_coeff=%.2f | "
                "min_pct=%.2f | usage_pct=%.2f",
                p_name,
                p_club,
                p_role,
                league_coeff,
                p_minutes_pct,
                p_usage_pct,
            )

        # Apply the formula for each metric
        for metric in _METRICS:
            pos_weight = POSITIONAL_WEIGHTS[metric][p_role]
            club_value = club_metrics[metric]

            contribution = club_value * (
                pos_weight * league_coeff * p_minutes_pct * p_usage_pct
            )

            team_totals[metric] += contribution

            if verbose and pos_weight > 0.0:
                logger.info(
                    "      %s: %.4f * (%.2f * %.2f * %.2f * %.2f) = %.5f",
                    metric.ljust(20),
                    club_value,
                    pos_weight,
                    league_coeff,
                    p_minutes_pct,
                    p_usage_pct,
                    contribution,
                )

    # ── Build output DataFrame ───────────────────────────────────────────────
    result = pd.DataFrame([{"team": team_name, **team_totals}])

    if verbose:
        logger.info("-" * 62)
        logger.info("Synthetic profile for %s:", team_name)
        for metric, value in team_totals.items():
            logger.info("  %-25s = %.5f", metric, value)
        logger.info("=" * 62)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 6. BATCH HELPER — synthesise multiple national teams at once
# ─────────────────────────────────────────────────────────────────────────────


def synthesize_multiple_teams(
    team_rosters: dict[str, list[dict[str, Any]]],
    club_metrics_df: pd.DataFrame,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Convenience wrapper to synthesise several national teams in one call.

    Parameters
    ----------
    team_rosters : dict[str, list[dict]]
        Keys are team names; values are 11-player roster_json lists.
    club_metrics_df : pd.DataFrame
        Shared club metrics DataFrame (same schema as synthesize_team).
    verbose : bool
        Passed through to synthesize_team for per-player logging.

    Returns
    -------
    pd.DataFrame
        Multi-row DataFrame, one synthesised profile per national team.
    """
    frames: list[pd.DataFrame] = []
    for team_name, roster in team_rosters.items():
        row = synthesize_team(
            roster_json=roster,
            club_metrics_df=club_metrics_df,
            team_name=team_name,
            verbose=verbose,
        )
        frames.append(row)

    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# 7. QUICK SMOKE TEST — runs when the module is executed directly
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Minimal stub roster (from the system context spec) ───────────────────
    roster_stub: list[dict[str, Any]] = [
        {
            "player_name": "Unai Simon",
            "club": "Athletic Club",
            "league": "La Liga",
            "role": "Goalkeeper",
            "minutes_played_pct": 0.95,
            "usage_rate_pct": 0.09,
        },
        {
            "player_name": "Dani Carvajal",
            "club": "Real Madrid",
            "league": "La Liga",
            "role": "Right Back",
            "minutes_played_pct": 0.80,
            "usage_rate_pct": 0.10,
        },
        {
            "player_name": "Aymeric Laporte",
            "club": "Al Nassr",
            "league": "Saudi Pro League",
            "role": "Center Back",
            "minutes_played_pct": 0.88,
            "usage_rate_pct": 0.09,
        },
        {
            "player_name": "Robin Le Normand",
            "club": "Atletico Madrid",
            "league": "La Liga",
            "role": "Center Back",
            "minutes_played_pct": 0.82,
            "usage_rate_pct": 0.09,
        },
        {
            "player_name": "Marc Cucurella",
            "club": "Chelsea",
            "league": "Premier League",
            "role": "Left Back",
            "minutes_played_pct": 0.78,
            "usage_rate_pct": 0.10,
        },
        {
            "player_name": "Rodri",
            "club": "Manchester City",
            "league": "Premier League",
            "role": "Defensive Midfield",
            "minutes_played_pct": 0.85,
            "usage_rate_pct": 0.12,
        },
        {
            "player_name": "Pedri",
            "club": "Barcelona",
            "league": "La Liga",
            "role": "Central Midfield",
            "minutes_played_pct": 0.72,
            "usage_rate_pct": 0.13,
        },
        {
            "player_name": "Fabián Ruiz",
            "club": "PSG",
            "league": "Ligue 1",
            "role": "Attacking Midfield",
            "minutes_played_pct": 0.80,
            "usage_rate_pct": 0.11,
        },
        {
            "player_name": "Lamine Yamal",
            "club": "Barcelona",
            "league": "La Liga",
            "role": "Right Wing",
            "minutes_played_pct": 0.90,
            "usage_rate_pct": 0.14,
        },
        {
            "player_name": "Nico Williams",
            "club": "Athletic Club",
            "league": "La Liga",
            "role": "Left Wing",
            "minutes_played_pct": 0.88,
            "usage_rate_pct": 0.13,
        },
        {
            "player_name": "Alvaro Morata",
            "club": "Atletico Madrid",
            "league": "La Liga",
            "role": "Striker",
            "minutes_played_pct": 0.70,
            "usage_rate_pct": 0.08,
        },
    ]

    # ── Stub club metrics (per-90 values, illustrative) ─────────────────────
    club_metrics_stub = pd.DataFrame(
        [
            {
                "club_name": "Athletic Club",
                "npxG": 1.42,
                "xT_Proxy": 8.1,
                "Deep_Progressions": 14.2,
                "PPDA": 8.5,
                "Shot_Suppression": 11.3,
            },
            {
                "club_name": "Real Madrid",
                "npxG": 1.98,
                "xT_Proxy": 10.3,
                "Deep_Progressions": 18.7,
                "PPDA": 10.2,
                "Shot_Suppression": 9.8,
            },
            {
                "club_name": "Al Nassr",
                "npxG": 1.10,
                "xT_Proxy": 6.4,
                "Deep_Progressions": 11.1,
                "PPDA": 13.5,
                "Shot_Suppression": 13.6,
            },
            {
                "club_name": "Atletico Madrid",
                "npxG": 1.35,
                "xT_Proxy": 7.6,
                "Deep_Progressions": 13.4,
                "PPDA": 9.1,
                "Shot_Suppression": 8.7,
            },
            {
                "club_name": "Chelsea",
                "npxG": 1.61,
                "xT_Proxy": 9.2,
                "Deep_Progressions": 16.5,
                "PPDA": 9.8,
                "Shot_Suppression": 12.1,
            },
            {
                "club_name": "Manchester City",
                "npxG": 2.10,
                "xT_Proxy": 11.5,
                "Deep_Progressions": 21.3,
                "PPDA": 7.4,
                "Shot_Suppression": 9.2,
            },
            {
                "club_name": "Barcelona",
                "npxG": 1.87,
                "xT_Proxy": 10.8,
                "Deep_Progressions": 19.8,
                "PPDA": 7.9,
                "Shot_Suppression": 10.4,
            },
            {
                "club_name": "PSG",
                "npxG": 1.75,
                "xT_Proxy": 9.9,
                "Deep_Progressions": 17.6,
                "PPDA": 8.6,
                "Shot_Suppression": 11.7,
            },
        ]
    )

    print("\n" + "=" * 62)
    print("SMOKE TEST: Spain 4-3-3 Synthetic Profile")
    print("=" * 62)

    result_df = synthesize_team(
        roster_json=roster_stub,
        club_metrics_df=club_metrics_stub,
        team_name="Spain",
        verbose=True,
    )

    print("\n─── Final Synthetic Team Row ───")
    print(result_df.to_string(index=False))

    # Sanity check: all metrics should be > 0
    metric_cols = [c for c in result_df.columns if c != "team"]
    assert all(result_df[metric_cols].iloc[0] > 0), (
        "Sanity check failed: some metrics are zero."
    )
    print("\n✅ Smoke test passed — all synthetic metrics are positive.")
