"""
Unit Tests for Baseline Engine.
Validates trailing 8-week rolling average (strictly prior weeks, no lookahead)
and peer route average (excluding self).
"""

import pytest
import pandas as pd
import numpy as np

from src.data_pipeline import get_processed_shipment_data
from src.baseline_engine import BaselineEngine
from src.config import settings


@pytest.fixture
def processed_data():
    return get_processed_shipment_data(settings.SHIPMENT_RECORDS_PATH)


def test_trailing_8_weeks_no_lookahead(processed_data):
    engine = BaselineEngine(trailing_weeks=8)
    df = engine.compute_baselines(processed_data)
    
    # Check Delhi-Jaipur on 2024-11-11
    target_route = "Delhi-Jaipur"
    target_week = pd.Timestamp("2024-11-11")
    
    row = df[(df["route"] == target_route) & (df["week_of"] == target_week)].iloc[0]
    
    # Manually compute prior 8 available weeks
    prior_weeks = processed_data[
        (processed_data["route"] == target_route) & (processed_data["week_of"] < target_week)
    ].tail(8)
    
    assert len(prior_weeks) == 8
    manual_avg = prior_weeks["cost_per_tonne_km"].mean()
    assert abs(row["own_history_avg_cptk"] - manual_avg) < 1e-6
    
    expected_pct = ((row["cost_per_tonne_km"] - manual_avg) / manual_avg) * 100.0
    assert row["vs_own_history"] == f"{expected_pct:+.1f}% vs this route's past average"
    assert row["vs_own_history"] == "+35.5% vs this route's past average"


def test_peer_routes_exclude_self(processed_data):
    engine = BaselineEngine(trailing_weeks=8)
    df = engine.compute_baselines(processed_data)
    
    # Check Delhi-Jaipur (Short) on 2024-11-11
    target_route = "Delhi-Jaipur"
    target_week = pd.Timestamp("2024-11-11")
    
    row = df[(df["route"] == target_route) & (df["week_of"] == target_week)].iloc[0]
    
    # Peer route for Short is Mumbai-Pune
    peers = processed_data[
        (processed_data["route_type"] == "Short")
        & (processed_data["week_of"] == target_week)
        & (processed_data["route"] != target_route)
    ]
    
    assert len(peers) == 1
    assert peers.iloc[0]["route"] == "Mumbai-Pune"
    
    peer_avg = peers["cost_per_tonne_km"].mean()
    assert abs(row["similar_routes_avg_cptk"] - peer_avg) < 1e-6
    
    expected_peer_pct = ((row["cost_per_tonne_km"] - peer_avg) / peer_avg) * 100.0
    assert row["vs_similar_routes"] == f"{expected_peer_pct:+.1f}% vs similar-length routes this week"
    assert row["vs_similar_routes"] == "+21.0% vs similar-length routes this week"


def test_first_week_handles_no_prior_history(processed_data):
    engine = BaselineEngine(trailing_weeks=8)
    df = engine.compute_baselines(processed_data)
    
    # First week of Delhi-Jaipur
    first_row = df[df["route"] == "Delhi-Jaipur"].iloc[0]
    assert np.isnan(first_row["own_history_avg_cptk"])
    assert first_row["vs_own_history"] == "N/A (first week)"
