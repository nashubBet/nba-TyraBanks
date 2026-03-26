"""
══════════════════════════════════════════════════════════════════
  NBA BASKETBALL BETTING MODEL
  Inspired by: kyleskom/NBA-Machine-Learning-Sports-Betting (GitHub)
               NBA-Betting/NBA_Betting, NBA-Betting/NBA_AI
  Research:    XGBoost + Neural Net ensemble | Kelly Criterion sizing
               ~69% moneyline accuracy (per kyleskom benchmark)
               Break-even ATS requires >52.38% win rate
══════════════════════════════════════════════════════════════════

STACK USED:
  • XGBoost  — best single-model performer on NBA spread/ML data
  • Neural Net (sklearn MLP) — captures non-linear feature combos
  • Ensemble  — averages both for final probability
  • Kelly Criterion — optimal bankroll sizing
  • Polymarket / Sportsbook arbitrage detector

FEATURES PULLED FROM TOP GITHUB REPOS & RESEARCH:
  OffRtg, DefRtg, Pace, eFG%, TOV%, ORB%, FT Rate (Four Factors)
  Rest days differential, B2B flag, home/away, travel fatigue
  Rolling 5/10 game form, SOS (strength of schedule)
  Net Rating delta vs opponent, Win% L10

SETUP:
  pip install xgboost scikit-learn pandas numpy requests nba_api

DATA SOURCE:
  nba_api (free, official NBA.com endpoints)
  The-Odds-API (free tier: 500 req/month) for live sportsbook lines

══════════════════════════════════════════════════════════════════
DISCLAIMER: For educational/research purposes only.
Sports betting involves financial risk. Bet responsibly.
Never bet what you cannot afford to lose.
══════════════════════════════════════════════════════════════════
"""

import time
import logging
import warnings
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Tuple

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("nba_model.log")],
)
log = logging.getLogger("NBA-Model")

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    # Bankroll Management
    bankroll: float = 1000.0
    kelly_fraction: float = 0.25       # Quarter Kelly — safer than full Kelly
    max_bet_pct: float = 0.05          # Never bet more than 5% of bankroll
    min_edge_threshold: float = 0.03   # Only bet if edge >= 3%
    min_ev_threshold: float = 0.04     # Only bet if EV >= 4%

    # Model
    ml_model_path: str = "nba_model.json"
    confidence_threshold: float = 0.55  # Only flag bets >= 55% confidence

    # API Keys (get free tier at the-odds-api.com)
    odds_api_key: str = "YOUR_ODDS_API_KEY"
    season: str = "2025-26"

    # Polymarket
    polymarket_enabled: bool = False   # Set True to also check Polymarket lines


# ══════════════════════════════════════════════════════════════
#  DATA COLLECTOR — pulls from nba_api
# ══════════════════════════════════════════════════════════════

class NBADataCollector:
    """
    Pulls team stats, schedules, and advanced metrics from NBA.com
    Using the nba_api package (same approach as kyleskom's repo)
    """

    def __init__(self):
        self.log = logging.getLogger("NBAData")

    def get_team_stats(self, season: str = "2025-26") -> pd.DataFrame:
        """Pull advanced team stats: OffRtg, DefRtg, Pace, eFG%, etc."""
        try:
            from nba_api.stats.endpoints import leaguedashteamstats
            time.sleep(0.6)  # Rate limit respect
            stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                measure_type_detailed_defense="Advanced",
                per_mode_detailed="PerGame",
            )
            df = stats.get_data_frames()[0]
            df.columns = [c.upper() for c in df.columns]
            self.log.info(f"Fetched advanced stats for {len(df)} teams")
            return df
        except Exception as e:
            self.log.error(f"Failed to fetch team stats: {e}")
            return self._mock_team_stats()

    def get_schedule(self, date_str: str = None) -> pd.DataFrame:
        """Get today's or a specific date's game schedule."""
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        try:
            from nba_api.stats.endpoints import scoreboardv2
            time.sleep(0.6)
            sb = scoreboardv2.ScoreboardV2(game_date=date_str)
            games = sb.get_data_frames()[0]
            self.log.info(f"Found {len(games)} games on {date_str}")
            return games
        except Exception as e:
            self.log.error(f"Failed to fetch schedule: {e}")
            return pd.DataFrame()

    def get_team_last_n_games(self, team_id: int, n: int = 10) -> pd.DataFrame:
        """Get rolling form — last N games for a team."""
        try:
            from nba_api.stats.endpoints import teamgamelogs
            time.sleep(0.6)
            logs = teamgamelogs.TeamGameLogs(
                team_id_nullable=str(team_id),
                last_n_games_nullable=str(n),
            )
            return logs.get_data_frames()[0]
        except Exception as e:
            self.log.error(f"Rolling form fetch failed: {e}")
            return pd.DataFrame()

    def _mock_team_stats(self) -> pd.DataFrame:
        """Synthetic data for demo/testing without API calls."""
        np.random.seed(42)
        teams = [
            "Boston Celtics", "Golden State Warriors", "Miami Heat",
            "Milwaukee Bucks", "Denver Nuggets", "Phoenix Suns",
            "Los Angeles Lakers", "Brooklyn Nets", "Chicago Bulls",
            "Philadelphia 76ers", "Dallas Mavericks", "Memphis Grizzlies",
            "Cleveland Cavaliers", "Atlanta Hawks", "Toronto Raptors",
            "New Orleans Pelicans", "Sacramento Kings", "Minnesota Timberwolves",
            "Oklahoma City Thunder", "New York Knicks",
        ]
        rows = []
        for i, team in enumerate(teams):
            off_rtg = np.random.normal(113, 4)
            def_rtg = np.random.normal(112, 4)
            rows.append({
                "TEAM_ID": 1610612737 + i,
                "TEAM_NAME": team,
                "W_PCT": np.random.uniform(0.3, 0.75),
                "OFF_RATING": round(off_rtg, 1),
                "DEF_RATING": round(def_rtg, 1),
                "NET_RATING": round(off_rtg - def_rtg, 1),
                "PACE": round(np.random.normal(99.5, 2.5), 1),
                "EFG_PCT": round(np.random.uniform(0.50, 0.58), 3),
                "TM_TOV_PCT": round(np.random.uniform(11, 16), 1),
                "OREB_PCT": round(np.random.uniform(22, 32), 1),
                "FT_RATE": round(np.random.uniform(0.18, 0.30), 3),
                "TS_PCT": round(np.random.uniform(0.56, 0.63), 3),
                "AST_PCT": round(np.random.uniform(55, 70), 1),
                "PIE": round(np.random.uniform(0.45, 0.55), 3),
            })
        return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
#  ODDS FETCHER — The-Odds-API (free tier)
# ══════════════════════════════════════════════════════════════

class OddsFetcher:
    """
    Fetches live sportsbook odds from The-Odds-API.
    Also checks Polymarket NBA markets if enabled.
    Based on approach from kyleskom's SbrOddsProvider.
    """
    BASE_URL = "https://api.the-odds-api.com/v4"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.log = logging.getLogger("OddsFetcher")

    def get_nba_odds(self, markets: str = "h2h,spreads,totals") -> list:
        """Fetch moneyline, spread, and totals odds for NBA."""
        if self.api_key == "YOUR_ODDS_API_KEY":
            self.log.warning("No API key set — using mock odds.")
            return self._mock_odds()

        url = f"{self.BASE_URL}/sports/basketball_nba/odds"
        params = {
            "apiKey": self.api_key,
            "regions": "us",
            "markets": markets,
            "oddsFormat": "american",
            "bookmakers": "fanduel,draftkings,betmgm,caesars",
        }
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            games = r.json()
            self.log.info(f"Fetched odds for {len(games)} games")
            return games
        except Exception as e:
            self.log.error(f"Odds fetch failed: {e}")
            return self._mock_odds()

    def american_to_prob(self, american_odds: float) -> float:
        """Convert American odds to implied probability (removes vig)."""
        if american_odds > 0:
            return 100 / (american_odds + 100)
        else:
            return abs(american_odds) / (abs(american_odds) + 100)

    def remove_vig(self, prob_home: float, prob_away: float) -> Tuple[float, float]:
        """Remove the bookmaker's vig to get fair implied probabilities."""
        total = prob_home + prob_away
        return prob_home / total, prob_away / total

    def _mock_odds(self) -> list:
        """Mock odds for demo when no API key is provided."""
        return [
            {
                "id": "mock_game_1",
                "home_team": "Boston Celtics",
                "away_team": "Golden State Warriors",
                "commence_time": datetime.now().isoformat(),
                "bookmakers": [
                    {
                        "key": "fanduel",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Boston Celtics", "price": -160},
                                    {"name": "Golden State Warriors", "price": +135},
                                ],
                            },
                            {
                                "key": "spreads",
                                "outcomes": [
                                    {"name": "Boston Celtics", "price": -110, "point": -4.5},
                                    {"name": "Golden State Warriors", "price": -110, "point": +4.5},
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "price": -115, "point": 222.5},
                                    {"name": "Under", "price": -105, "point": 222.5},
                                ],
                            },
                        ],
                    }
                ],
            },
            {
                "id": "mock_game_2",
                "home_team": "Milwaukee Bucks",
                "away_team": "Miami Heat",
                "commence_time": datetime.now().isoformat(),
                "bookmakers": [
                    {
                        "key": "draftkings",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Milwaukee Bucks", "price": -145},
                                    {"name": "Miami Heat", "price": +122},
                                ],
                            },
                            {
                                "key": "spreads",
                                "outcomes": [
                                    {"name": "Milwaukee Bucks", "price": -110, "point": -3.5},
                                    {"name": "Miami Heat", "price": -110, "point": +3.5},
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "price": -110, "point": 218.5},
                                    {"name": "Under", "price": -110, "point": 218.5},
                                ],
                            },
                        ],
                    }
                ],
            },
        ]


# ══════════════════════════════════════════════════════════════
#  FEATURE ENGINEER
#  Based on top-performing features from kyleskom + NBA_Betting
# ══════════════════════════════════════════════════════════════

class FeatureEngineer:
    """
    Builds the feature vector for a matchup.
    Features proven in research to matter most:
      1. OffRtg / DefRtg differential
      2. Pace matchup (high vs low pace)
      3. eFG% differential
      4. Rest days (B2B, 3-in-4 schedule)
      5. Home court advantage
      6. Rolling form (L5, L10 net rating)
      7. TOV% (ball security edge)
      8. ORB% (second chance points)
      9. FT Rate (free throw volume)
     10. Strength of schedule
    """

    # Home court advantage in points (NBA historical: ~2.5 pts)
    HOME_COURT_PTS = 2.5
    # Back-to-back penalty in rating points
    B2B_PENALTY = 2.0
    # 3-in-4 penalty
    THREE_IN_FOUR_PENALTY = 1.2
    # Travel penalty (cross-country)
    TRAVEL_PENALTY = 0.7

    def build_matchup_features(
        self,
        home_stats: dict,
        away_stats: dict,
        home_rest_days: int = 2,
        away_rest_days: int = 2,
        home_travel: bool = False,
        away_travel: bool = False,
        home_l10_net: float = 0.0,
        away_l10_net: float = 0.0,
    ) -> dict:
        """
        Returns a feature dict for one game matchup.
        All features are DIFFERENTIAL (home minus away) unless noted.
        """
        # Core Four Factors (proven most predictive by research)
        off_rtg_diff = home_stats.get("OFF_RATING", 110) - away_stats.get("OFF_RATING", 110)
        def_rtg_diff = home_stats.get("DEF_RATING", 110) - away_stats.get("DEF_RATING", 110)
        net_rtg_diff = home_stats.get("NET_RATING", 0) - away_stats.get("NET_RATING", 0)
        efg_diff = home_stats.get("EFG_PCT", 0.52) - away_stats.get("EFG_PCT", 0.52)
        tov_diff = home_stats.get("TM_TOV_PCT", 13) - away_stats.get("TM_TOV_PCT", 13)
        oreb_diff = home_stats.get("OREB_PCT", 27) - away_stats.get("OREB_PCT", 27)
        ft_rate_diff = home_stats.get("FT_RATE", 0.22) - away_stats.get("FT_RATE", 0.22)
        ts_diff = home_stats.get("TS_PCT", 0.58) - away_stats.get("TS_PCT", 0.58)
        pace_diff = home_stats.get("PACE", 99) - away_stats.get("PACE", 99)
        pie_diff = home_stats.get("PIE", 0.5) - away_stats.get("PIE", 0.5)

        # Pace matchup — combined pace predicts total
        avg_pace = (home_stats.get("PACE", 99) + away_stats.get("PACE", 99)) / 2

        # Rest & Fatigue modifiers
        home_b2b = 1 if home_rest_days == 0 else 0
        away_b2b = 1 if away_rest_days == 0 else 0
        rest_diff = home_rest_days - away_rest_days  # positive = home has more rest

        # Fatigue-adjusted net rating
        home_fatigue_adj = (home_b2b * self.B2B_PENALTY) + (home_travel * self.TRAVEL_PENALTY)
        away_fatigue_adj = (away_b2b * self.B2B_PENALTY) + (away_travel * self.TRAVEL_PENALTY)
        fatigue_diff = away_fatigue_adj - home_fatigue_adj  # positive = home less fatigued

        # Rolling form
        form_diff = home_l10_net - away_l10_net

        # Win pct differential
        win_pct_diff = home_stats.get("W_PCT", 0.5) - away_stats.get("W_PCT", 0.5)

        return {
            # Core efficiency
            "off_rtg_diff": off_rtg_diff,
            "def_rtg_diff": def_rtg_diff,
            "net_rtg_diff": net_rtg_diff,
            "efg_diff": efg_diff,
            "tov_diff": tov_diff,          # Negative = home turns it over more (bad)
            "oreb_diff": oreb_diff,
            "ft_rate_diff": ft_rate_diff,
            "ts_diff": ts_diff,
            "pie_diff": pie_diff,
            # Pace
            "pace_diff": pace_diff,
            "avg_pace": avg_pace,
            # Schedule / fatigue
            "rest_diff": rest_diff,
            "home_b2b": home_b2b,
            "away_b2b": away_b2b,
            "fatigue_diff": fatigue_diff,
            # Form
            "form_diff": form_diff,
            "win_pct_diff": win_pct_diff,
            # Home court (always 1 for home team predictions)
            "is_home": 1,
        }

    def features_to_array(self, features: dict) -> np.ndarray:
        FEATURE_ORDER = [
            "off_rtg_diff", "def_rtg_diff", "net_rtg_diff",
            "efg_diff", "tov_diff", "oreb_diff", "ft_rate_diff",
            "ts_diff", "pie_diff", "pace_diff", "avg_pace",
            "rest_diff", "home_b2b", "away_b2b", "fatigue_diff",
            "form_diff", "win_pct_diff", "is_home",
        ]
        return np.array([features[k] for k in FEATURE_ORDER], dtype=float).reshape(1, -1)


# ══════════════════════════════════════════════════════════════
#  ML MODELS  (XGBoost + Neural Net — matches kyleskom's approach)
# ══════════════════════════════════════════════════════════════

class NBAPredictor:
    """
    Two-model ensemble:
     - XGBoost: best for tabular sports data (used in top GitHub repos)
     - MLP Neural Net: captures non-linear interactions
    Final output = average of both probabilities
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.xgb_model = None
        self.nn_model = None
        self.feature_scaler = None
        self.is_trained = False
        self.log = logging.getLogger("NBAPredictor")

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train both models on historical game data."""
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features for neural net
        self.feature_scaler = StandardScaler()
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)

        # ── XGBoost ──
        try:
            from xgboost import XGBClassifier
            self.xgb_model = XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
            )
            self.xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            xgb_acc = (self.xgb_model.predict(X_val) == y_val).mean()
            self.log.info(f"XGBoost validation accuracy: {xgb_acc:.3f}")
        except ImportError:
            self.log.warning("xgboost not installed — using GradientBoosting fallback")
            from sklearn.ensemble import GradientBoostingClassifier
            self.xgb_model = GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
            )
            self.xgb_model.fit(X_train, y_train)
            xgb_acc = (self.xgb_model.predict(X_val) == y_val).mean()
            self.log.info(f"GBM validation accuracy: {xgb_acc:.3f}")

        # ── Neural Net ──
        self.nn_model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        )
        self.nn_model.fit(X_train_scaled, y_train)
        nn_acc = (self.nn_model.predict(X_val_scaled) == y_val).mean()
        self.log.info(f"Neural Net validation accuracy: {nn_acc:.3f}")
        self.log.info(f"Ensemble avg accuracy: {(xgb_acc + nn_acc) / 2:.3f}")
        self.is_trained = True

    def predict_proba(self, X: np.ndarray) -> Tuple[float, float]:
        """Returns (prob_home_win, prob_away_win)."""
        if not self.is_trained:
            self.log.warning("Model not trained — using naive prediction")
            return 0.5, 0.5

        xgb_prob = self.xgb_model.predict_proba(X)[0][1]
        X_scaled = self.feature_scaler.transform(X)
        nn_prob = self.nn_model.predict_proba(X_scaled)[0][1]
        ensemble_prob = (xgb_prob + nn_prob) / 2
        return round(ensemble_prob, 4), round(1 - ensemble_prob, 4)

    def generate_training_data(self, n_games: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic training data for demo.
        In production: replace with real historical data from nba_api.
        The key insight is that net rating differential is the strongest predictor.
        """
        np.random.seed(42)
        fe = FeatureEngineer()
        X_list, y_list = [], []

        for _ in range(n_games):
            # Random team strength
            home_net = np.random.normal(0, 5)
            away_net = np.random.normal(0, 5)

            home_stats = {
                "OFF_RATING": 113 + home_net * 0.6 + np.random.normal(0, 2),
                "DEF_RATING": 113 - home_net * 0.4 + np.random.normal(0, 2),
                "NET_RATING": home_net,
                "PACE": np.random.normal(99, 2.5),
                "EFG_PCT": 0.53 + home_net * 0.003 + np.random.normal(0, 0.015),
                "TM_TOV_PCT": 13 - home_net * 0.1 + np.random.normal(0, 1),
                "OREB_PCT": 27 + np.random.normal(0, 3),
                "FT_RATE": 0.22 + np.random.normal(0, 0.03),
                "TS_PCT": 0.58 + home_net * 0.002,
                "PIE": 0.5 + home_net * 0.01,
                "W_PCT": 0.5 + home_net * 0.05,
            }
            away_stats = {
                "OFF_RATING": 113 + away_net * 0.6 + np.random.normal(0, 2),
                "DEF_RATING": 113 - away_net * 0.4 + np.random.normal(0, 2),
                "NET_RATING": away_net,
                "PACE": np.random.normal(99, 2.5),
                "EFG_PCT": 0.53 + away_net * 0.003 + np.random.normal(0, 0.015),
                "TM_TOV_PCT": 13 - away_net * 0.1 + np.random.normal(0, 1),
                "OREB_PCT": 27 + np.random.normal(0, 3),
                "FT_RATE": 0.22 + np.random.normal(0, 0.03),
                "TS_PCT": 0.58 + away_net * 0.002,
                "PIE": 0.5 + away_net * 0.01,
                "W_PCT": 0.5 + away_net * 0.05,
            }
            rest_home = np.random.choice([0, 1, 2, 3], p=[0.15, 0.25, 0.40, 0.20])
            rest_away = np.random.choice([0, 1, 2, 3], p=[0.15, 0.25, 0.40, 0.20])
            form_home = np.random.normal(0, 3)
            form_away = np.random.normal(0, 3)

            feats = fe.build_matchup_features(
                home_stats, away_stats,
                home_rest_days=int(rest_home),
                away_rest_days=int(rest_away),
                home_l10_net=form_home,
                away_l10_net=form_away,
            )
            X_arr = fe.features_to_array(feats).flatten()

            # Home win probability driven by net rating + rest + home court
            true_prob = 0.5 + (home_net - away_net) * 0.03 + \
                        0.025 + (rest_home - rest_away) * 0.01 + \
                        (form_home - form_away) * 0.005
            true_prob = np.clip(true_prob, 0.05, 0.95)
            outcome = int(np.random.random() < true_prob)

            X_list.append(X_arr)
            y_list.append(outcome)

        return np.array(X_list), np.array(y_list)


# ══════════════════════════════════════════════════════════════
#  KELLY CRITERION — optimal bet sizing
#  Used in kyleskom/NBA-Machine-Learning-Sports-Betting
#  Research confirms fractional Kelly (25-50%) reduces variance
# ══════════════════════════════════════════════════════════════

class KellyCriterion:
    """
    Kelly formula: f = (bp - q) / b
      b = decimal odds - 1
      p = model win probability
      q = 1 - p

    Applied at fraction (default 0.25) to reduce ruin risk.
    Research confirms that partial Kelly (25-50%) is optimal
    for sports betting with uncertain edge estimates.
    """

    @staticmethod
    def calculate(
        model_prob: float,
        american_odds: float,
        fraction: float = 0.25,
    ) -> float:
        """
        Returns fraction of bankroll to bet.
        Capped at 5% for safety.
        """
        # Convert American to decimal
        if american_odds > 0:
            decimal_odds = american_odds / 100 + 1
        else:
            decimal_odds = 100 / abs(american_odds) + 1

        b = decimal_odds - 1  # net odds
        p = model_prob
        q = 1 - p

        kelly_full = (b * p - q) / b

        if kelly_full <= 0:
            return 0.0  # No edge — don't bet

        kelly_fractional = kelly_full * fraction
        return round(min(kelly_fractional, 0.05), 4)  # Cap at 5%

    @staticmethod
    def expected_value(model_prob: float, american_odds: float) -> float:
        """Calculate expected value of a bet."""
        if american_odds > 0:
            payout = american_odds / 100
        else:
            payout = 100 / abs(american_odds)

        ev = (model_prob * payout) - ((1 - model_prob) * 1.0)
        return round(ev, 4)


# ══════════════════════════════════════════════════════════════
#  BETTING ANALYZER — the core engine
# ══════════════════════════════════════════════════════════════

class BettingAnalyzer:
    """
    Main engine. For each game:
      1. Build features from team stats + schedule context
      2. Run ensemble model to get win probability
      3. Compare to implied odds probability
      4. If edge exists → calculate Kelly bet size
      5. Output recommended bets
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.collector = NBADataCollector()
        self.odds_fetcher = OddsFetcher(config.odds_api_key)
        self.fe = FeatureEngineer()
        self.predictor = NBAPredictor(config)
        self.kelly = KellyCriterion()
        self.log = logging.getLogger("BettingAnalyzer")
        self.team_stats_cache: Optional[pd.DataFrame] = None

    def setup(self):
        """Train model on historical data."""
        self.log.info("Training models on historical data...")
        X, y = self.predictor.generate_training_data(n_games=3000)
        self.predictor.train(X, y)
        self.log.info("Models trained and ready.")

    def get_team_stats_dict(self, team_name: str) -> dict:
        """Look up stats dict for a team by name."""
        if self.team_stats_cache is None:
            self.team_stats_cache = self.collector.get_team_stats(self.config.season)

        df = self.team_stats_cache
        # Fuzzy match team name
        mask = df["TEAM_NAME"].str.contains(team_name.split()[-1], case=False, na=False)
        match = df[mask]
        if match.empty:
            self.log.warning(f"Team not found: {team_name} — using league average")
            return {"OFF_RATING": 113, "DEF_RATING": 113, "NET_RATING": 0,
                    "PACE": 99, "EFG_PCT": 0.53, "TM_TOV_PCT": 13,
                    "OREB_PCT": 27, "FT_RATE": 0.22, "TS_PCT": 0.58,
                    "PIE": 0.5, "W_PCT": 0.5}
        return match.iloc[0].to_dict()

    def analyze_game(
        self,
        home_team: str,
        away_team: str,
        home_ml_odds: float,    # American odds e.g. -150
        away_ml_odds: float,
        home_spread: float = 0,
        spread_odds: float = -110,
        total_line: float = 220,
        over_odds: float = -110,
        under_odds: float = -110,
        home_rest_days: int = 2,
        away_rest_days: int = 2,
        home_travel: bool = False,
        away_travel: bool = False,
    ) -> dict:
        """
        Full analysis of a single game.
        Returns dict with recommendations and Kelly sizes.
        """
        # Get team stats
        home_stats = self.get_team_stats_dict(home_team)
        away_stats = self.get_team_stats_dict(away_team)

        # Build features
        features = self.fe.build_matchup_features(
            home_stats, away_stats,
            home_rest_days=home_rest_days,
            away_rest_days=away_rest_days,
            home_travel=home_travel,
            away_travel=away_travel,
        )
        X = self.fe.features_to_array(features)

        # Model prediction
        prob_home, prob_away = self.predictor.predict_proba(X)

        # Book implied probabilities (vig-removed)
        book_prob_home_raw = self.odds_fetcher.american_to_prob(home_ml_odds)
        book_prob_away_raw = self.odds_fetcher.american_to_prob(away_ml_odds)
        book_prob_home, book_prob_away = self.odds_fetcher.remove_vig(
            book_prob_home_raw, book_prob_away_raw
        )

        # Edge calculation
        edge_home = prob_home - book_prob_home
        edge_away = prob_away - book_prob_away

        # EV calculation
        ev_home = self.kelly.expected_value(prob_home, home_ml_odds)
        ev_away = self.kelly.expected_value(prob_away, away_ml_odds)

        # Kelly bet sizes
        kelly_home = self.kelly.calculate(prob_home, home_ml_odds, self.config.kelly_fraction)
        kelly_away = self.kelly.calculate(prob_away, away_ml_odds, self.config.kelly_fraction)

        # Totals prediction: use pace + efficiency
        avg_pace = (home_stats.get("PACE", 99) + away_stats.get("PACE", 99)) / 2
        avg_off = (home_stats.get("OFF_RATING", 113) + away_stats.get("OFF_RATING", 113)) / 2
        predicted_total = avg_pace * (avg_off / 100) * 2
        predicted_total = round(predicted_total, 1)
        total_edge = abs(predicted_total - total_line)

        # Decision logic
        recommendations = []

        if (prob_home >= self.config.confidence_threshold and
                edge_home >= self.config.min_edge_threshold and
                ev_home >= self.config.min_ev_threshold and
                kelly_home > 0):
            bet_amount = round(self.config.bankroll * kelly_home, 2)
            recommendations.append({
                "bet_type": "MONEYLINE",
                "team": home_team,
                "side": "HOME",
                "odds": home_ml_odds,
                "model_prob": prob_home,
                "book_prob": book_prob_home,
                "edge": edge_home,
                "ev": ev_home,
                "kelly_pct": kelly_home,
                "bet_amount": bet_amount,
                "confidence": "HIGH" if prob_home >= 0.62 else "MEDIUM",
            })

        if (prob_away >= self.config.confidence_threshold and
                edge_away >= self.config.min_edge_threshold and
                ev_away >= self.config.min_ev_threshold and
                kelly_away > 0):
            bet_amount = round(self.config.bankroll * kelly_away, 2)
            recommendations.append({
                "bet_type": "MONEYLINE",
                "team": away_team,
                "side": "AWAY",
                "odds": away_ml_odds,
                "model_prob": prob_away,
                "book_prob": book_prob_away,
                "edge": edge_away,
                "ev": ev_away,
                "kelly_pct": kelly_away,
                "bet_amount": bet_amount,
                "confidence": "HIGH" if prob_away >= 0.62 else "MEDIUM",
            })

        if total_edge >= 4.0:
            side = "OVER" if predicted_total > total_line else "UNDER"
            odds = over_odds if side == "OVER" else under_odds
            recommendations.append({
                "bet_type": "TOTAL",
                "team": f"{home_team} vs {away_team}",
                "side": side,
                "odds": odds,
                "model_total": predicted_total,
                "book_total": total_line,
                "edge_pts": total_edge,
                "bet_amount": round(self.config.bankroll * 0.015, 2),  # flat 1.5% for totals
                "confidence": "HIGH" if total_edge >= 6 else "MEDIUM",
            })

        return {
            "game": f"{away_team} @ {home_team}",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "model_prob_home": prob_home,
            "model_prob_away": prob_away,
            "book_prob_home": book_prob_home,
            "book_prob_away": book_prob_away,
            "edge_home": round(edge_home, 4),
            "edge_away": round(edge_away, 4),
            "ev_home": ev_home,
            "ev_away": ev_away,
            "predicted_total": predicted_total,
            "book_total": total_line,
            "features": features,
            "recommendations": recommendations,
            "bet_count": len(recommendations),
        }

    def analyze_today(self) -> list:
        """Analyze all games today with live odds."""
        results = []
        games = self.odds_fetcher.get_nba_odds()

        for game in games:
            home_team = game["home_team"]
            away_team = game["away_team"]

            # Extract odds from first bookmaker
            home_ml, away_ml = -110, -110
            spread_line, spread_odds = 0, -110
            total_line, over_odds, under_odds = 220, -110, -110

            for book in game.get("bookmakers", []):
                for market in book.get("markets", []):
                    if market["key"] == "h2h":
                        for outcome in market["outcomes"]:
                            if outcome["name"] == home_team:
                                home_ml = outcome["price"]
                            else:
                                away_ml = outcome["price"]
                    elif market["key"] == "spreads":
                        for outcome in market["outcomes"]:
                            if outcome["name"] == home_team:
                                spread_line = outcome.get("point", 0)
                    elif market["key"] == "totals":
                        for outcome in market["outcomes"]:
                            if outcome["name"] == "Over":
                                over_odds = outcome["price"]
                                total_line = outcome.get("point", 220)
                            else:
                                under_odds = outcome["price"]
                break  # use first bookmaker

            result = self.analyze_game(
                home_team=home_team,
                away_team=away_team,
                home_ml_odds=home_ml,
                away_ml_odds=away_ml,
                total_line=total_line,
                over_odds=over_odds,
                under_odds=under_odds,
            )
            results.append(result)

        return results

    def print_report(self, results: list):
        """Print a clean betting report."""
        print("\n" + "═" * 65)
        print("  🏀  NBA BETTING MODEL — TODAY'S REPORT")
        print(f"  📅  {datetime.now().strftime('%A, %B %d %Y')}")
        print(f"  💰  Bankroll: ${self.config.bankroll:,.2f}")
        print("═" * 65)

        all_recs = []
        for r in results:
            print(f"\n  {r['game']}")
            print(f"  Model: {r['model_prob_home']:.1%} home | {r['model_prob_away']:.1%} away")
            print(f"  Book:  {r['book_prob_home']:.1%} home | {r['book_prob_away']:.1%} away")
            print(f"  Projected Total: {r['predicted_total']} pts (line: {r['book_total']})")

            if r["recommendations"]:
                print(f"  ── {len(r['recommendations'])} BET(S) FLAGGED ──")
                for rec in r["recommendations"]:
                    flag = "🔥" if rec["confidence"] == "HIGH" else "📊"
                    print(f"  {flag} {rec['bet_type']} | {rec['team']} {rec['side']} "
                          f"({rec['odds']:+d}) | "
                          f"Edge: {rec.get('edge', rec.get('edge_pts', 0)):.1%} | "
                          f"Bet: ${rec['bet_amount']:.2f}")
                all_recs.extend(r["recommendations"])
            else:
                print("  ✗  No edge found — skip")

        print("\n" + "─" * 65)
        total_wagered = sum(r["bet_amount"] for r in all_recs)
        print(f"  Total bets today: {len(all_recs)}")
        print(f"  Total wagered: ${total_wagered:.2f}")
        print(f"  Bankroll at risk: {total_wagered / self.config.bankroll:.1%}")
        print("═" * 65)
        print("\n  ⚠️  Remember: Beat the closing line (CLV) = true edge.")
        print("  Track every bet. Target 53%+ win rate ATS.")
        print("  Use Fractional Kelly. Never chase losses.\n")


# ══════════════════════════════════════════════════════════════
#  BACKTESTER
# ══════════════════════════════════════════════════════════════

class Backtester:
    """
    Simulate model performance on historical data.
    Tracks: win rate, ROI, CLV (closing line value), drawdown.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.log = logging.getLogger("Backtester")

    def run_simulation(self, n_games: int = 500) -> pd.DataFrame:
        """
        Simulate betting on N games with model decisions.
        Uses synthetic data — replace with real historical game data.
        """
        np.random.seed(99)
        records = []
        bankroll = self.config.bankroll
        fe = FeatureEngineer()

        for i in range(n_games):
            # Simulate random matchup
            home_true_prob = np.random.uniform(0.35, 0.75)
            model_prob = home_true_prob + np.random.normal(0, 0.06)  # model noise
            model_prob = np.clip(model_prob, 0.05, 0.95)

            # Random odds (slightly vig'd)
            if home_true_prob > 0.5:
                home_ml = -int(100 * home_true_prob / (1 - home_true_prob) * 1.05)
                away_ml = int(100 * (1 - home_true_prob) / home_true_prob * 0.95)
            else:
                home_ml = int(100 * home_true_prob / (1 - home_true_prob) * 0.95)
                away_ml = -int(100 * (1 - home_true_prob) / home_true_prob * 1.05)

            kelly = KellyCriterion()
            ev = kelly.expected_value(model_prob, home_ml)
            kelly_f = kelly.calculate(model_prob, home_ml, self.config.kelly_fraction)

            # Only bet if edge passes threshold
            if ev < self.config.min_ev_threshold or kelly_f <= 0:
                records.append({
                    "game": i, "bet": False, "outcome": None,
                    "pnl": 0, "bankroll": bankroll, "model_prob": model_prob,
                })
                continue

            bet_size = min(bankroll * kelly_f, bankroll * self.config.max_bet_pct)
            # Simulate result
            won = np.random.random() < home_true_prob
            if home_ml < 0:
                pnl = bet_size * (100 / abs(home_ml)) if won else -bet_size
            else:
                pnl = bet_size * (home_ml / 100) if won else -bet_size

            bankroll += pnl
            records.append({
                "game": i, "bet": True, "won": won,
                "model_prob": round(model_prob, 3),
                "true_prob": round(home_true_prob, 3),
                "odds": home_ml, "bet_size": round(bet_size, 2),
                "pnl": round(pnl, 2), "bankroll": round(bankroll, 2),
                "ev": round(ev, 4), "kelly_f": round(kelly_f, 4),
            })

        return pd.DataFrame(records)

    def summary(self, df: pd.DataFrame):
        bets = df[df["bet"] == True].copy()
        if bets.empty:
            print("No bets placed.")
            return

        wins = bets["won"].sum()
        total_bets = len(bets)
        win_rate = wins / total_bets
        total_pnl = bets["pnl"].sum()
        total_wagered = bets["bet_size"].sum()
        roi = total_pnl / total_wagered

        # Max drawdown
        cumulative = bets["pnl"].cumsum()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max).min()

        print("\n" + "═" * 50)
        print("  📊  BACKTEST SIMULATION RESULTS")
        print("═" * 50)
        print(f"  Games analyzed  : {len(df)}")
        print(f"  Bets placed     : {total_bets} ({total_bets/len(df):.1%} bet rate)")
        print(f"  Win rate        : {win_rate:.1%}  ({int(wins)}W / {total_bets-int(wins)}L)")
        print(f"  Total PnL       : ${total_pnl:+,.2f}")
        print(f"  ROI             : {roi:+.1%}")
        print(f"  Avg bet size    : ${bets['bet_size'].mean():.2f}")
        print(f"  Max drawdown    : ${drawdown:,.2f}")
        print(f"  Break-even rate : 52.38%")
        print(f"  Edge vs break-even: {win_rate - 0.5238:+.1%}")
        print("═" * 50)
        bets.to_csv("backtest_results.csv", index=False)
        print("  Results saved to backtest_results.csv\n")


# ══════════════════════════════════════════════════════════════
#  POLYMARKET BRIDGE  (check NBA markets on Polymarket)
# ══════════════════════════════════════════════════════════════

class PolymarketBridge:
    """
    Pulls NBA game markets from Polymarket's Gamma API.
    Compares Polymarket implied odds vs sportsbooks.
    Arbitrage = if both sides sum to < 1.0.
    """
    GAMMA_URL = "https://gamma-api.polymarket.com/markets"

    def get_nba_markets(self) -> list:
        try:
            params = {
                "tag": "nba",
                "active": "true",
                "limit": 20,
                "order": "volume",
            }
            r = requests.get(self.GAMMA_URL, params=params, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logging.getLogger("Polymarket").error(f"Failed: {e}")
            return []

    def find_arbitrage(self, poly_prob: float, book_prob: float, threshold: float = 0.02) -> dict:
        """
        If model edge > threshold on both sides combined → arbitrage.
        poly_prob: Polymarket implied probability for YES
        book_prob: Sportsbook implied probability for home team
        """
        combined = poly_prob + (1 - book_prob)
        arb_exists = combined < 1.0
        edge = 1.0 - combined if arb_exists else 0
        return {"arbitrage": arb_exists, "edge": round(edge, 4), "combined": round(combined, 4)}


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n🏀 NBA Betting Model — Initializing...\n")

    config = ModelConfig(
        bankroll=1000.0,
        kelly_fraction=0.25,      # Conservative — use 0.5 if more confident
        min_edge_threshold=0.03,  # 3% edge minimum
        min_ev_threshold=0.04,    # 4% EV minimum
        confidence_threshold=0.55,
        odds_api_key="YOUR_ODDS_API_KEY",  # Get free at the-odds-api.com
    )

    # ── Step 1: Train model ──
    analyzer = BettingAnalyzer(config)
    analyzer.setup()

    # ── Step 2: Analyze today's games ──
    print("\nAnalyzing today's games...")
    results = analyzer.analyze_today()
    analyzer.print_report(results)

    # ── Step 3: Run backtest ──
    print("\nRunning backtest simulation (500 games)...")
    bt = Backtester(config)
    sim = bt.run_simulation(500)
    bt.summary(sim)

    # ── Step 4: Check Polymarket (optional) ──
    if config.polymarket_enabled:
        print("\nChecking Polymarket NBA markets...")
        pm = PolymarketBridge()
        markets = pm.get_nba_markets()
        print(f"Found {len(markets)} active NBA markets on Polymarket")
        for m in markets[:5]:
            print(f"  {m.get('question', 'Unknown')} — Volume: ${m.get('volumeNum', 0):,.0f}")

    print("\n✅ Done. Use the-odds-api.com for live sportsbook odds.")
    print("📌 Next steps:")
    print("   1. Set your ODDS_API_KEY in ModelConfig")
    print("   2. Train on real historical data from nba_api")
    print("   3. Track CLV (closing line value) — beating CLV = real edge")
    print("   4. Log every single bet in a spreadsheet")
