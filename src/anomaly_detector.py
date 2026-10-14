"""
Anomaly Detector Module.
Detects when shipping cost is rising and looks out of the ordinary compared to
its own history or to similar routes.
"""

from typing import Optional
import pandas as pd
import numpy as np
from src.config import settings


class AnomalyDetector:
    """Identifies candidate anomalous route-weeks based on cost inflation."""

    def __init__(
        self,
        own_history_threshold_pct: float = settings.OWN_HISTORY_THRESHOLD_PCT,
        similar_routes_threshold_pct: float = settings.SIMILAR_ROUTES_THRESHOLD_PCT,
    ):
        self.own_threshold = own_history_threshold_pct
        self.peer_threshold = similar_routes_threshold_pct

    def is_anomalous_cost_rise(self, row: pd.Series) -> bool:
        """
        Determines whether a route-week constitutes an out-of-the-ordinary cost rise.
        Condition:
        - Cost is rising: vs_own_history_pct is positive (or vs_similar_routes_pct is elevated when own history is limited)
        - Looks out of ordinary: exceeds either own history threshold or similar routes threshold.
        """
        own_pct = row.get("vs_own_history_pct", np.nan)
        peer_pct = row.get("vs_similar_routes_pct", np.nan)
        
        # Check own history spike
        spike_vs_own = not np.isnan(own_pct) and own_pct >= self.own_threshold
        
        # Check peer route spike (cost is rising compared to peers, and not negative vs own)
        spike_vs_peer = not np.isnan(peer_pct) and peer_pct >= self.peer_threshold and (np.isnan(own_pct) or own_pct > 0)
        
        return bool(spike_vs_own or spike_vs_peer)

    def detect_anomalies(self, df_with_baselines: pd.DataFrame) -> pd.DataFrame:
        """
        Scans DataFrame with computed baselines and adds:
        - is_cost_rising_anomaly (bool): True if candidate anomaly requiring AI justification check
        """
        df = df_with_baselines.copy()
        df["is_cost_rising_anomaly"] = df.apply(self.is_anomalous_cost_rise, axis=1)
        return df


def detect_cost_anomalies(
    df_with_baselines: pd.DataFrame,
    own_threshold: float = settings.OWN_HISTORY_THRESHOLD_PCT,
    peer_threshold: float = settings.SIMILAR_ROUTES_THRESHOLD_PCT,
) -> pd.DataFrame:
    """Convenience helper to detect cost anomalies."""
    detector = AnomalyDetector(own_threshold, peer_threshold)
    return detector.detect_anomalies(df_with_baselines)
