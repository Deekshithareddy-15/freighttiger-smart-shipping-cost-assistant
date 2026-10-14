"""
Unit Tests for Data Pipeline.
Validates shipment grouping, Monday calendar week derivation, and CPTK formula.
"""

import pytest
import pandas as pd
from pathlib import Path

from src.data_pipeline import DataPipeline, get_processed_shipment_data
from src.config import settings


def test_shipment_loading_and_columns():
    pipeline = DataPipeline(settings.SHIPMENT_RECORDS_PATH)
    raw_df = pipeline.load_raw_data()
    assert not raw_df.empty
    assert len(raw_df) == 2940
    assert "shipment_id" in raw_df.columns
    assert "quantity_tonnes" in raw_df.columns
    assert "distance_km" in raw_df.columns
    assert "freight_cost_inr" in raw_df.columns


def test_monday_week_derivation():
    pipeline = DataPipeline(settings.SHIPMENT_RECORDS_PATH)
    grouped = pipeline.process()
    
    # Check all week_of are Mondays (weekday == 0)
    for w in grouped["week_of"].unique():
        ts = pd.Timestamp(w)
        assert ts.weekday() == 0, f"Week {w} is not a Monday!"


def test_cptk_formula():
    """
    Verifies:
    cost_per_tonne_km = total freight cost / (quantity carried in tonnes x distance travelled in km)
    """
    pipeline = DataPipeline(settings.SHIPMENT_RECORDS_PATH)
    grouped = pipeline.process()
    
    # Spot-check Delhi-Jaipur on 2024-11-11
    dj = grouped[(grouped["route"] == "Delhi-Jaipur") & (grouped["week_of"] == "2024-11-11")]
    assert len(dj) == 1
    dj_row = dj.iloc[0]
    
    expected_cptk = dj_row["total_freight_cost"] / dj_row["total_tonne_km"]
    assert abs(dj_row["cost_per_tonne_km"] - expected_cptk) < 1e-9
    assert round(dj_row["cost_per_tonne_km"], 2) == 4.17
