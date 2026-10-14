"""
Baseline Engine Module.
Calculates comparison baselines strictly matching FreightTiger case study definitions:
1. vs. own history: trailing 8-week rolling average CPTK using only weeks strictly
   prior to current week (no lookahead; uses all available prior weeks if < 8).
2. vs. similar routes: same-week average CPTK across all other routes sharing the same route_type
   (excludes the route itself from peer average).
"""

from typing import Optional, Tuple
import pandas as pd
import numpy as np


class BaselineEngine:
    """Computes own-history and peer-route baseline comparisons."""

    def __init__(self, trailing_weeks: int = 8):
        self.trailing_weeks = trailing_weeks

    def compute_baselines(self, route_weeks_df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes the aggregated route-week DataFrame from data_pipeline and adds:
        - own_history_avg_cptk (float)
        - own_history_prior_weeks_count (int)
        - vs_own_history_pct (float)
        - vs_own_history (str): formatted as "+XX.X% vs this route's past average"
        - similar_routes_avg_cptk (float)
        - similar_routes_peers_count (int)
        - vs_similar_routes_pct (float)
        - vs_similar_routes (str): formatted as "+XX.X% vs similar-length routes this week"
        """
        df = route_weeks_df.copy()
        
        # Ensure chronological order
        df = df.sort_values(["route", "week_of"]).reset_index(drop=True)
        
        own_avgs = []
        own_counts = []
        own_pcts = []
        own_strings = []
        
        peer_avgs = []
        peer_counts = []
        peer_pcts = []
        peer_strings = []
        
        for idx, row in df.iterrows():
            current_route = row["route"]
            current_week = row["week_of"]
            current_route_type = row["route_type"]
            current_cptk = row["cost_per_tonne_km"]
            
            # --- 1. vs. own history ---
            # Weeks strictly before current week (no look-ahead), up to self.trailing_weeks
            prior_weeks = df[
                (df["route"] == current_route) & (df["week_of"] < current_week)
            ].tail(self.trailing_weeks)
            
            prior_count = len(prior_weeks)
            if prior_count > 0:
                # Unweighted arithmetic mean of prior available weekly CPTKs
                own_avg = prior_weeks["cost_per_tonne_km"].mean()
                own_pct = ((current_cptk - own_avg) / own_avg) * 100.0
                own_str = f"{own_pct:+.1f}% vs this route's past average"
            else:
                own_avg = np.nan
                own_pct = np.nan
                own_str = "N/A (first week)"
                
            own_avgs.append(own_avg)
            own_counts.append(prior_count)
            own_pcts.append(own_pct)
            own_strings.append(own_str)
            
            # --- 2. vs. similar routes ---
            # Same week, same route_type, excluding the route itself
            peers = df[
                (df["route_type"] == current_route_type)
                & (df["week_of"] == current_week)
                & (df["route"] != current_route)
            ]
            
            peer_count = len(peers)
            if peer_count > 0:
                # Unweighted arithmetic mean of peer routes in that week
                peer_avg = peers["cost_per_tonne_km"].mean()
                peer_pct = ((current_cptk - peer_avg) / peer_avg) * 100.0
                peer_str = f"{peer_pct:+.1f}% vs similar-length routes this week"
            else:
                peer_avg = np.nan
                peer_pct = np.nan
                peer_str = "N/A (no peer routes)"
                
            peer_avgs.append(peer_avg)
            peer_counts.append(peer_count)
            peer_pcts.append(peer_pct)
            peer_strings.append(peer_str)
            
        df["own_history_avg_cptk"] = own_avgs
        df["own_history_prior_weeks_count"] = own_counts
        df["vs_own_history_pct"] = own_pcts
        df["vs_own_history"] = own_strings
        
        df["similar_routes_avg_cptk"] = peer_avgs
        df["similar_routes_peers_count"] = peer_counts
        df["vs_similar_routes_pct"] = peer_pcts
        df["vs_similar_routes"] = peer_strings
        
        return df


def compute_all_baselines(route_weeks_df: pd.DataFrame, trailing_weeks: int = 8) -> pd.DataFrame:
    """Convenience helper to compute all baselines."""
    engine = BaselineEngine(trailing_weeks=trailing_weeks)
    return engine.compute_baselines(route_weeks_df)
