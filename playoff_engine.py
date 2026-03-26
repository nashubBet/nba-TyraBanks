"""
══════════════════════════════════════════════════════════════════
  NBA PLAYOFF BETTING INTELLIGENCE ENGINE
  Plugs into nba_betting_model.py as an additional layer
  
  Philosophy:
  - Matchup-driven edges over star power / narratives
  - Playoff halfcourt fit > regular season pace stats
  - Coaching adjustments > raw talent
  - Market skepticism toward public favorites
  - Line movement + sharp money > public %
══════════════════════════════════════════════════════════════════
"""

import os
import json
import requests
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import logging

log = logging.getLogger("PlayoffEngine")

# ══════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════

@dataclass
class TeamPlayoffProfile:
    """Full playoff profile for one team."""
    name: str

    # 1. Baseline Strength
    off_rtg: float = 113.0
    def_rtg: float = 113.0
    net_rtg: float = 0.0
    pace: float = 99.0
    halfcourt_off_rtg: float = 110.0    # Slower pace, more halfcourt sets
    halfcourt_def_rtg: float = 110.0
    clutch_net_rtg: float = 0.0         # Net rating in last 5 min, within 5 pts
    recent_form_net: float = 0.0        # L5 game net rating average
    home_net_rtg: float = 3.0
    away_net_rtg: float = -3.0

    # 2. Matchup Factors
    pnr_offense_rating: float = 5.0     # 1-10: how good is their PnR attack
    pnr_defense_rating: float = 5.0     # 1-10: how good is their PnR coverage
    rim_pressure: float = 5.0           # drives + lobs per 100 possessions (normalized)
    rim_protection: float = 5.0         # shots defended at rim (normalized)
    three_pt_volume: float = 35.0       # 3PA per game
    three_pt_pct: float = 0.36
    opp_three_suppression: float = 5.0  # how well they limit 3PA (1-10)
    off_reb_pct: float = 27.0
    small_ball_vulnerability: float = 5.0
    turnover_pressure: float = 5.0      # forced TO rate (1-10)
    ballhandling_quality: float = 5.0   # turnover avoidance (1-10)
    bench_net_rtg: float = 0.0          # bench unit net rating

    # 3. Coach / Scheme
    scheme: str = "Switch"              # Switch / Drop / Blitz / Zone / Help
    adjustability: float = 5.0         # 1-10: willingness to adjust after G1
    rotation_depth: int = 9             # how many players get real minutes
    star_stagger_quality: float = 5.0   # 1-10: how well they manage star minutes
    defensive_versatility: float = 5.0  # can they guard multiple positions

    # 4. Market
    opening_spread: float = 0.0
    current_spread: float = 0.0
    opening_total: float = 220.0
    current_total: float = 220.0
    public_bet_pct: float = 50.0        # % of bets on this team
    public_money_pct: float = 50.0      # % of money on this team
    is_public_favorite: bool = False    # narrative-driven public darling


@dataclass
class PlayoffMatchupAnalysis:
    """Full output for one playoff game/series."""
    game: str
    home_team: str
    away_team: str
    date: str

    # Fair lines
    fair_spread: float = 0.0            # negative = home favored
    fair_total: float = 220.0

    # Best bets
    best_side: Optional[str] = None     # "HOME", "AWAY", or None
    best_side_team: Optional[str] = None
    best_total: Optional[str] = None    # "OVER", "UNDER", or None
    confidence_grade: str = "C"         # A+, A, B+, B, C, D

    # Analysis
    matchup_reasons: list = field(default_factory=list)   # 3 reasons
    market_reasons: list = field(default_factory=list)    # 2 reasons
    underpriced_factor: str = ""                          # 1 thing public misses

    # Raw numbers
    model_home_prob: float = 0.5
    edge_on_side: float = 0.0
    edge_on_total: float = 0.0


# ══════════════════════════════════════════════════════════════
#  PLAYOFF MATCHUP ANALYZER
# ══════════════════════════════════════════════════════════════

class PlayoffMatchupAnalyzer:
    """
    Core engine. Analyzes a playoff matchup across all 5 dimensions
    and returns a structured PlayoffMatchupAnalysis.
    """

    # Spread standard deviation for NBA playoffs (tighter than regular season)
    PLAYOFF_SIGMA = 10.5
    # Home court in playoffs (slightly less than regular season)
    PLAYOFF_HOME_COURT = 2.0
    # Public team spread penalty (they get bet up ~1-1.5 pts)
    PUBLIC_TEAM_PENALTY = 1.2

    def __init__(self):
        self.log = logging.getLogger("PlayoffMatchupAnalyzer")

    def analyze(
        self,
        home: TeamPlayoffProfile,
        away: TeamPlayoffProfile,
        game_label: str = "",
    ) -> PlayoffMatchupAnalysis:

        result = PlayoffMatchupAnalysis(
            game=game_label or f"{away.name} @ {home.name}",
            home_team=home.name,
            away_team=away.name,
            date=datetime.now().strftime("%Y-%m-%d"),
        )

        # ── 1. Fair Spread ──────────────────────────────────────
        result.fair_spread = self._calc_fair_spread(home, away)
        result.fair_total = self._calc_fair_total(home, away)

        # ── 2. Win Probability ──────────────────────────────────
        result.model_home_prob = self._win_prob(result.fair_spread)

        # ── 3. Matchup Mismatches ───────────────────────────────
        result.matchup_reasons = self._find_matchup_edges(home, away)

        # ── 4. Market Analysis ──────────────────────────────────
        result.market_reasons, side_edge, total_edge = self._market_analysis(
            home, away, result.fair_spread, result.fair_total
        )
        result.edge_on_side = side_edge
        result.edge_on_total = total_edge

        # ── 5. Best Bet Decision ────────────────────────────────
        result.best_side, result.best_side_team = self._best_side(
            home, away, result.fair_spread, side_edge
        )
        result.best_total = self._best_total(total_edge)

        # ── 6. Underpriced Factor ───────────────────────────────
        result.underpriced_factor = self._find_underpriced(home, away)

        # ── 7. Confidence Grade ─────────────────────────────────
        result.confidence_grade = self._grade(
            abs(side_edge), abs(total_edge), len(result.matchup_reasons)
        )

        return result

    # ── Fair Spread ─────────────────────────────────────────────

    def _calc_fair_spread(self, home: TeamPlayoffProfile, away: TeamPlayoffProfile) -> float:
        """
        Build fair spread from:
        - Net rating differential (weighted toward halfcourt in playoffs)
        - Recent form
        - Clutch performance
        - Home court
        - Public team adjustment
        """
        # Blend season net with halfcourt net (playoffs = more halfcourt)
        home_adj_net = (home.net_rtg * 0.4) + \
                       ((home.halfcourt_off_rtg - home.halfcourt_def_rtg) * 0.6)
        away_adj_net = (away.net_rtg * 0.4) + \
                       ((away.halfcourt_off_rtg - away.halfcourt_def_rtg) * 0.6)

        net_diff = home_adj_net - away_adj_net

        # Recent form adjustment (last 5 games)
        form_adj = (home.recent_form_net - away.recent_form_net) * 0.15

        # Clutch adjustment (playoffs are often decided in clutch)
        clutch_adj = (home.clutch_net_rtg - away.clutch_net_rtg) * 0.2

        # Home court
        home_court = self.PLAYOFF_HOME_COURT

        # Raw spread (negative = home favored)
        raw_spread = -(net_diff * 0.4 + form_adj + clutch_adj + home_court)

        # Adjust if away team is a public favorite (market inflates them)
        if away.is_public_favorite:
            raw_spread -= self.PUBLIC_TEAM_PENALTY  # pushes spread toward home
        if home.is_public_favorite:
            raw_spread += self.PUBLIC_TEAM_PENALTY

        return round(raw_spread, 1)

    def _calc_fair_total(self, home: TeamPlayoffProfile, away: TeamPlayoffProfile) -> float:
        """
        Playoff totals are driven by halfcourt pace and defensive intensity.
        Typically 5-8 pts lower than regular season equivalent.
        """
        avg_off = (home.halfcourt_off_rtg + away.halfcourt_off_rtg) / 2
        avg_def = (home.halfcourt_def_rtg + away.halfcourt_def_rtg) / 2
        avg_pace = (home.pace + away.pace) / 2 * 0.94  # Playoff pace ~6% slower

        # Points per possession estimate
        ppp = ((avg_off / 100) + (1 - avg_def / 120)) / 2
        raw_total = avg_pace * ppp * 2
        return round(raw_total, 1)

    def _win_prob(self, fair_spread: float) -> float:
        """Convert fair spread to win probability."""
        from scipy.stats import norm
        try:
            prob = norm.cdf(-fair_spread / self.PLAYOFF_SIGMA)
        except Exception:
            prob = 0.5 + (-fair_spread / self.PLAYOFF_SIGMA) * 0.08
        return round(max(0.05, min(0.95, prob)), 3)

    # ── Matchup Edges ───────────────────────────────────────────

    def _find_matchup_edges(self, home: TeamPlayoffProfile, away: TeamPlayoffProfile) -> list:
        """
        Find the 3 most important matchup mismatches.
        Returns plain English reasons.
        """
        edges = []

        # PnR mismatch
        pnr_diff = away.pnr_offense_rating - home.pnr_defense_rating
        if abs(pnr_diff) >= 2:
            if pnr_diff > 0:
                edges.append({
                    "category": "PnR Mismatch",
                    "edge_team": away.name,
                    "detail": f"{away.name}'s PnR attack ({away.pnr_offense_rating:.0f}/10) "
                              f"exploits {home.name}'s coverage weakness "
                              f"({home.pnr_defense_rating:.0f}/10). "
                              f"Expect higher usage for ball handler in pick situations.",
                    "magnitude": abs(pnr_diff),
                })
            else:
                edges.append({
                    "category": "PnR Defense",
                    "edge_team": home.name,
                    "detail": f"{home.name}'s PnR defense ({home.pnr_defense_rating:.0f}/10) "
                              f"neutralizes {away.name}'s primary action "
                              f"(off rating: {away.pnr_offense_rating:.0f}/10). "
                              f"Watch for early offensive struggles.",
                    "magnitude": abs(pnr_diff),
                })

        # Rim pressure vs protection
        rim_diff = away.rim_pressure - home.rim_protection
        if abs(rim_diff) >= 2:
            if rim_diff > 0:
                edges.append({
                    "category": "Rim Pressure",
                    "edge_team": away.name,
                    "detail": f"{away.name} attacks the rim aggressively (pressure: "
                              f"{away.rim_pressure:.0f}/10) vs {home.name}'s weak rim "
                              f"protection ({home.rim_protection:.0f}/10). "
                              f"Drives, lobs, and FT volume favors {away.name}.",
                    "magnitude": abs(rim_diff),
                })
            else:
                edges.append({
                    "category": "Rim Protection",
                    "edge_team": home.name,
                    "detail": f"{home.name}'s rim protection ({home.rim_protection:.0f}/10) "
                              f"walls off {away.name}'s drive game. "
                              f"Forces {away.name} into uncomfortable midrange.",
                    "magnitude": abs(rim_diff),
                })

        # 3-point profile vs suppression
        three_pt_edge = away.three_pt_volume * away.three_pt_pct - \
                        (10 - home.opp_three_suppression) * 2
        if abs(three_pt_edge) >= 2:
            if three_pt_edge > 0:
                edges.append({
                    "category": "3PT Volume Edge",
                    "edge_team": away.name,
                    "detail": f"{away.name} generates high 3PT volume "
                              f"({away.three_pt_volume:.0f} attempts/game at "
                              f"{away.three_pt_pct:.1%}) against a team that allows "
                              f"open looks (suppression: {home.opp_three_suppression:.0f}/10).",
                    "magnitude": abs(three_pt_edge),
                })
            else:
                edges.append({
                    "category": "3PT Suppression",
                    "edge_team": home.name,
                    "detail": f"{home.name} limits 3PT attempts effectively "
                              f"(suppression: {home.opp_three_suppression:.0f}/10), "
                              f"forcing {away.name} into a more difficult offensive diet.",
                    "magnitude": abs(three_pt_edge),
                })

        # Offensive rebounding vs small-ball
        oreb_edge = home.off_reb_pct - (10 - away.small_ball_vulnerability) * 2.5
        if oreb_edge >= 3:
            edges.append({
                "category": "OREB Mismatch",
                "edge_team": home.name,
                "detail": f"{home.name}'s offensive rebounding ({home.off_reb_pct:.0f}%) "
                          f"vs {away.name}'s small-ball vulnerability "
                          f"({away.small_ball_vulnerability:.0f}/10 risk). "
                          f"Second-chance points could be decisive.",
                "magnitude": oreb_edge,
            })

        # Turnover pressure vs ballhandling
        to_edge = home.turnover_pressure - away.ballhandling_quality
        if abs(to_edge) >= 2:
            if to_edge > 0:
                edges.append({
                    "category": "Turnover Pressure",
                    "edge_team": home.name,
                    "detail": f"{home.name} applies high turnover pressure "
                              f"({home.turnover_pressure:.0f}/10) against "
                              f"{away.name}'s shaky ballhandling "
                              f"({away.ballhandling_quality:.0f}/10). "
                              f"Live-ball turnovers = easy transition points.",
                    "magnitude": abs(to_edge),
                })
            else:
                edges.append({
                    "category": "Ball Security",
                    "edge_team": away.name,
                    "detail": f"{away.name}'s disciplined ballhandling "
                              f"({away.ballhandling_quality:.0f}/10) neutralizes "
                              f"{home.name}'s pressure defense. "
                              f"Limits transition opportunities for home team.",
                    "magnitude": abs(to_edge),
                })

        # Bench depth
        bench_diff = home.bench_net_rtg - away.bench_net_rtg
        if abs(bench_diff) >= 3:
            better = home.name if bench_diff > 0 else away.name
            edges.append({
                "category": "Bench Advantage",
                "edge_team": better,
                "detail": f"{better}'s bench unit outperforms by "
                          f"{abs(bench_diff):.1f} net rating points. "
                          f"In a 7-game series, bench minutes compound — "
                          f"especially when starters need rest in Q3.",
                "magnitude": abs(bench_diff),
            })

        # Sort by magnitude, return top 3
        edges.sort(key=lambda x: x["magnitude"], reverse=True)
        return edges[:3]

    # ── Market Analysis ─────────────────────────────────────────

    def _market_analysis(
        self,
        home: TeamPlayoffProfile,
        away: TeamPlayoffProfile,
        fair_spread: float,
        fair_total: float,
    ) -> tuple:
        """
        Returns (market_reasons list, side_edge float, total_edge float).
        Looks for line movement, public % discrepancies, sharp money signals.
        """
        reasons = []
        side_edge = 0.0
        total_edge = 0.0

        # Side: compare fair spread to current market spread
        # Use away team's current spread as reference
        market_spread = away.current_spread  # negative = away favored
        spread_edge = fair_spread - market_spread
        side_edge = spread_edge

        # Line movement direction
        spread_move = away.current_spread - away.opening_spread
        if abs(spread_move) >= 1.0:
            direction = "moved toward" if spread_move < 0 else "moved away from"
            sharp = "sharp money likely on" if spread_move < 0 else "public pushing"
            reasons.append({
                "category": "Line Movement",
                "detail": f"Spread has moved {abs(spread_move):.1f} pts — "
                          f"{direction} {away.name}. {sharp} {away.name}. "
                          f"Movement of 1.5+ pts in playoffs = significant signal.",
                "signal": "SHARP" if abs(spread_move) >= 1.5 else "MODERATE",
            })

        # Public % vs money % discrepancy (reverse line movement signal)
        public_diff = away.public_bet_pct - away.public_money_pct
        if abs(public_diff) >= 10:
            if public_diff > 0:
                reasons.append({
                    "category": "Sharp vs Public Split",
                    "detail": f"{away.name} getting {away.public_bet_pct:.0f}% of bets "
                              f"but only {away.public_money_pct:.0f}% of money. "
                              f"Sharp/large bettors disagree with public — "
                              f"fade signal on {away.name}.",
                    "signal": "SHARP",
                })
            else:
                reasons.append({
                    "category": "Sharp Money",
                    "detail": f"{away.name} getting {away.public_bet_pct:.0f}% of bets "
                              f"but {away.public_money_pct:.0f}% of money. "
                              f"Bigger bets going on {away.name} — sharp action.",
                    "signal": "SHARP",
                })

        # Public narrative inflation
        if away.is_public_favorite:
            reasons.append({
                "category": "Public Narrative Inflation",
                "detail": f"{away.name} is a heavy public team — media narrative, "
                          f"star power, and brand driving {away.public_bet_pct:.0f}% "
                          f"of bets. Books shade the line {self.PUBLIC_TEAM_PENALTY} pts "
                          f"against them. Historical fade rate on public playoff "
                          f"favorites ATS is above 54%.",
                "signal": "FADE",
            })
        elif home.is_public_favorite:
            reasons.append({
                "category": "Public Narrative Inflation",
                "detail": f"{home.name} is a public darling at home. "
                          f"Home crowd narrative inflating the spread. "
                          f"Look for away team to cover if matchup edges favor them.",
                "signal": "FADE",
            })

        # Totals: fair vs market
        total_move = away.current_total - away.opening_total
        total_edge = fair_total - away.current_total
        if abs(total_move) >= 2:
            reasons.append({
                "category": "Total Line Movement",
                "detail": f"Total moved {abs(total_move):.1f} pts "
                          f"({'up' if total_move > 0 else 'down'}) from open. "
                          f"Model fair total: {fair_total:.1f}. "
                          f"{'Injury news or pace concern priced in.' if total_move < 0 else 'Public over action inflating total.'}",
                "signal": "UNDER" if total_move > 0 else "OVER",
            })

        # Spread edge vs fair value
        if abs(spread_edge) >= 1.5:
            edge_team = away.name if spread_edge < 0 else home.name
            reasons.append({
                "category": "Fair Value Gap",
                "detail": f"Model fair spread: {fair_spread:+.1f}. "
                          f"Market spread: {market_spread:+.1f}. "
                          f"Gap of {abs(spread_edge):.1f} pts — "
                          f"{edge_team} has value at current number.",
                "signal": "VALUE",
            })

        return reasons[:2], side_edge, total_edge

    # ── Best Bet Decisions ──────────────────────────────────────

    def _best_side(
        self,
        home: TeamPlayoffProfile,
        away: TeamPlayoffProfile,
        fair_spread: float,
        side_edge: float,
    ) -> tuple:
        MIN_EDGE = 1.5  # minimum spread edge to recommend
        if abs(side_edge) < MIN_EDGE:
            return None, None
        if side_edge < 0:  # fair spread more negative than market = away value
            return "AWAY", away.name
        else:
            return "HOME", home.name

    def _best_total(self, total_edge: float) -> Optional[str]:
        MIN_TOTAL_EDGE = 2.5
        if abs(total_edge) < MIN_TOTAL_EDGE:
            return None
        return "OVER" if total_edge > 0 else "UNDER"

    def _find_underpriced(self, home: TeamPlayoffProfile, away: TeamPlayoffProfile) -> str:
        """
        Identify the single most underpriced factor the public/sportsbook misses.
        """
        candidates = []

        # Coaching adjustability gap
        adj_diff = abs(home.adjustability - away.adjustability)
        if adj_diff >= 2:
            better_coach = home.name if home.adjustability > away.adjustability else away.name
            candidates.append((adj_diff, f"{better_coach}'s coaching adjustability "
                f"({max(home.adjustability, away.adjustability):.0f}/10) — "
                f"in a series, Game 2-7 adjustments matter more than Game 1 "
                f"execution. Public models Game 1 too heavily."))

        # Clutch net rating
        clutch_diff = abs(home.clutch_net_rtg - away.clutch_net_rtg)
        if clutch_diff >= 4:
            better_clutch = home.name if home.clutch_net_rtg > away.clutch_net_rtg else away.name
            candidates.append((clutch_diff, f"{better_clutch}'s clutch net rating "
                f"advantage ({max(home.clutch_net_rtg, away.clutch_net_rtg):+.1f}) — "
                f"playoff games are decided in the final 5 minutes. "
                f"Public bets on overall talent, not late-game execution."))

        # Bench depth
        bench_diff = abs(home.bench_net_rtg - away.bench_net_rtg)
        if bench_diff >= 3:
            better_bench = home.name if home.bench_net_rtg > away.bench_net_rtg else away.name
            candidates.append((bench_diff, f"{better_bench}'s bench depth "
                f"(+{bench_diff:.1f} bench net rtg) — books and public focus "
                f"on starters, but bench units decide 3rd quarter runs "
                f"and foul trouble situations."))

        # Halfcourt vs pace gap
        hc_diff = abs(
            (home.halfcourt_off_rtg - home.halfcourt_def_rtg) -
            (away.halfcourt_off_rtg - away.halfcourt_def_rtg)
        )
        if hc_diff >= 3:
            better_hc = home.name if (home.halfcourt_off_rtg - home.halfcourt_def_rtg) > \
                        (away.halfcourt_off_rtg - away.halfcourt_def_rtg) else away.name
            candidates.append((hc_diff, f"{better_hc}'s halfcourt efficiency gap "
                f"(+{hc_diff:.1f} pts) — public uses regular season pace stats "
                f"but playoffs slow down 15-20% into halfcourt sets. "
                f"This team is built for it."))

        if not candidates:
            return ("Rotation shortening — playoff rotations shrink to 8-9 players. "
                    "The team with a stronger top-8 outperforms their regular season "
                    "numbers when the coach tightens the bench.")

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def _grade(self, side_edge: float, total_edge: float, matchup_count: int) -> str:
        score = side_edge * 0.5 + total_edge * 0.3 + matchup_count * 0.5
        if score >= 5: return "A+"
        if score >= 4: return "A"
        if score >= 3: return "B+"
        if score >= 2: return "B"
        if score >= 1: return "C"
        return "D"


# ══════════════════════════════════════════════════════════════
#  PRINT REPORT
# ══════════════════════════════════════════════════════════════

def print_playoff_report(analysis: PlayoffMatchupAnalysis):
    grade_colors = {"A+": "🔥", "A": "✅", "B+": "📊", "B": "📊", "C": "⚠️", "D": "❌"}
    icon = grade_colors.get(analysis.confidence_grade, "📊")

    print("\n" + "═" * 65)
    print(f"  🏀 PLAYOFF INTELLIGENCE REPORT")
    print(f"  {analysis.game}  —  {analysis.date}")
    print("═" * 65)

    print(f"\n  {icon} CONFIDENCE GRADE: {analysis.confidence_grade}")
    print(f"  Fair Spread : {analysis.fair_spread:+.1f}  |  Fair Total: {analysis.fair_total:.1f}")
    print(f"  Model Home Win Prob: {analysis.model_home_prob:.1%}")

    if analysis.best_side:
        print(f"\n  ✅ BEST SIDE  : {analysis.best_side_team} ({analysis.best_side})")
        print(f"     Spread Edge : {analysis.edge_on_side:+.1f} pts vs market")
    else:
        print(f"\n  ⛔ SIDE       : No edge — skip or reduce size")

    if analysis.best_total:
        print(f"  ✅ BEST TOTAL : {analysis.best_total}")
        print(f"     Total Edge  : {analysis.edge_on_total:+.1f} pts vs market")
    else:
        print(f"  ⛔ TOTAL      : No edge on total")

    print(f"\n  ── 3 MATCHUP REASONS ──────────────────────────────")
    for i, r in enumerate(analysis.matchup_reasons, 1):
        print(f"  {i}. [{r['category']}] {r['edge_team']}")
        print(f"     {r['detail']}")

    print(f"\n  ── 2 MARKET REASONS ───────────────────────────────")
    for i, r in enumerate(analysis.market_reasons, 1):
        print(f"  {i}. [{r['category']}] Signal: {r.get('signal','—')}")
        print(f"     {r['detail']}")

    print(f"\n  ── 1 UNDERPRICED FACTOR ───────────────────────────")
    print(f"  💡 {analysis.underpriced_factor}")
    print("═" * 65)


# ══════════════════════════════════════════════════════════════
#  FLASK ROUTES — add to app.py
# ══════════════════════════════════════════════════════════════

def register_playoff_routes(app, config):
    """
    Call this from app.py to add playoff endpoints.
    Usage in app.py:
        from playoff_engine import register_playoff_routes
        register_playoff_routes(app, config)
    """
    from flask import request, jsonify

    analyzer = PlayoffMatchupAnalyzer()

    @app.route("/api/playoff", methods=["POST"])
    def playoff_analysis():
        """
        POST body example:
        {
          "home": { "name": "Boston Celtics", "net_rtg": 8.2, ... },
          "away": { "name": "Miami Heat", "net_rtg": 2.1, ... }
        }
        """
        data = request.json or {}
        home_data = data.get("home", {})
        away_data = data.get("away", {})

        home = TeamPlayoffProfile(**{
            k: v for k, v in home_data.items()
            if k in TeamPlayoffProfile.__dataclass_fields__
        })
        away = TeamPlayoffProfile(**{
            k: v for k, v in away_data.items()
            if k in TeamPlayoffProfile.__dataclass_fields__
        })

        result = analyzer.analyze(home, away)
        return jsonify({
            "game": result.game,
            "fair_spread": result.fair_spread,
            "fair_total": result.fair_total,
            "best_side": result.best_side,
            "best_side_team": result.best_side_team,
            "best_total": result.best_total,
            "confidence_grade": result.confidence_grade,
            "model_home_prob": result.model_home_prob,
            "edge_on_side": result.edge_on_side,
            "edge_on_total": result.edge_on_total,
            "matchup_reasons": result.matchup_reasons,
            "market_reasons": result.market_reasons,
            "underpriced_factor": result.underpriced_factor,
        })

    @app.route("/api/playoff/demo")
    def playoff_demo():
        """Returns a sample playoff analysis with demo data."""
        home = TeamPlayoffProfile(
            name="Boston Celtics",
            net_rtg=8.2, off_rtg=119.4, def_rtg=111.2,
            halfcourt_off_rtg=116.0, halfcourt_def_rtg=109.0,
            pace=99.1, clutch_net_rtg=6.2, recent_form_net=5.1,
            pnr_defense_rating=8.0, rim_protection=7.5,
            opp_three_suppression=7.0, turnover_pressure=7.5,
            bench_net_rtg=2.1, adjustability=8.0,
            scheme="Switch", rotation_depth=9,
            opening_spread=-5.5, current_spread=-5.5,
            opening_total=214.5, current_total=212.5,
            public_bet_pct=62, public_money_pct=70,
            is_public_favorite=True,
        )
        away = TeamPlayoffProfile(
            name="Miami Heat",
            net_rtg=1.8, off_rtg=113.2, def_rtg=111.4,
            halfcourt_off_rtg=111.5, halfcourt_def_rtg=108.0,
            pace=97.2, clutch_net_rtg=4.8, recent_form_net=3.2,
            pnr_offense_rating=7.5, rim_pressure=6.5,
            three_pt_volume=38, three_pt_pct=0.37,
            ballhandling_quality=7.0, bench_net_rtg=1.4,
            adjustability=9.0, scheme="Zone",
            opening_spread=5.5, current_spread=4.0,
            opening_total=214.5, current_total=212.5,
            public_bet_pct=38, public_money_pct=30,
            is_public_favorite=False,
        )
        result = analyzer.analyze(home, away, "Miami Heat @ Boston Celtics — ECF G1")
        return jsonify({
            "game": result.game,
            "fair_spread": result.fair_spread,
            "fair_total": result.fair_total,
            "best_side": result.best_side,
            "best_side_team": result.best_side_team,
            "best_total": result.best_total,
            "confidence_grade": result.confidence_grade,
            "model_home_prob": result.model_home_prob,
            "matchup_reasons": result.matchup_reasons,
            "market_reasons": result.market_reasons,
            "underpriced_factor": result.underpriced_factor,
        })


# ══════════════════════════════════════════════════════════════
#  DEMO — run standalone
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    analyzer = PlayoffMatchupAnalyzer()

    # Example: ECF Game 1
    home = TeamPlayoffProfile(
        name="Boston Celtics",
        net_rtg=8.2, off_rtg=119.4, def_rtg=111.2,
        halfcourt_off_rtg=116.0, halfcourt_def_rtg=109.0,
        pace=99.1, clutch_net_rtg=6.2, recent_form_net=5.1,
        pnr_defense_rating=8.0, rim_protection=7.5,
        opp_three_suppression=7.0, turnover_pressure=7.5,
        bench_net_rtg=2.1, adjustability=8.0,
        scheme="Switch", rotation_depth=9,
        opening_spread=-5.5, current_spread=-5.5,
        opening_total=214.5, current_total=212.5,
        public_bet_pct=62, public_money_pct=70,
        is_public_favorite=True,
    )
    away = TeamPlayoffProfile(
        name="Miami Heat",
        net_rtg=1.8, off_rtg=113.2, def_rtg=111.4,
        halfcourt_off_rtg=111.5, halfcourt_def_rtg=108.0,
        pace=97.2, clutch_net_rtg=4.8, recent_form_net=3.2,
        pnr_offense_rating=7.5, rim_pressure=6.5,
        three_pt_volume=38, three_pt_pct=0.37,
        ballhandling_quality=7.0, bench_net_rtg=1.4,
        adjustability=9.0, scheme="Zone",
        opening_spread=5.5, current_spread=4.0,
        opening_total=214.5, current_total=212.5,
        public_bet_pct=38, public_money_pct=30,
        is_public_favorite=False,
    )

    result = analyzer.analyze(home, away, "Miami Heat @ Boston Celtics — ECF G1")
    print_playoff_report(result)
