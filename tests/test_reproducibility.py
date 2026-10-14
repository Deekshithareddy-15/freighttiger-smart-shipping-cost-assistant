"""
Reproducibility Tests.
Verifies that 3 untouched runs on the same dataset produce identical outputs:
identical flags, identical CPTK numbers, identical comparison percentages, and identical note citations.
"""

import pytest
import pandas as pd
from src.assistant import ShippingCostAssistant


def test_three_untouched_runs_match():
    assistant = ShippingCostAssistant()
    
    # Run 1
    df1, _ = assistant.run_pipeline(export_csv=False)
    
    # Run 2
    df2, _ = assistant.run_pipeline(export_csv=False)
    
    # Run 3
    df3, _ = assistant.run_pipeline(export_csv=False)
    
    # 1. Exact row count match
    assert len(df1) == len(df2) == len(df3), "Run row counts do not match!"
    
    # 2. Strict columns match
    cols_to_check = [
        "route",
        "week_of",
        "cost_per_tonne_km",
        "vs_own_history",
        "vs_similar_routes",
        "flagged",
        "matched_note_id"
    ]
    
    pd.testing.assert_frame_equal(df1[cols_to_check], df2[cols_to_check], check_dtype=True)
    pd.testing.assert_frame_equal(df2[cols_to_check], df3[cols_to_check], check_dtype=True)
    
    # 3. Explanations should also be identical for deterministic reasoner
    pd.testing.assert_series_equal(df1["reason"], df2["reason"])
    pd.testing.assert_series_equal(df2["reason"], df3["reason"])
