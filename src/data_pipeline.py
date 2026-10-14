"""
Data Pipeline Module.
Handles ingestion, schema validation, route identification, Monday-aligned weekly grouping,
and calculation of Cost Per Tonne Per Km (CPTK).
"""

from pathlib import Path
from typing import Union
import pandas as pd
import numpy as np


REQUIRED_COLUMNS = [
    "shipment_id",
    "origin",
    "destination",
    "route_type",
    "material",
    "quantity_tonnes",
    "distance_km",
    "freight_cost_inr",
    "shipment_date",
    "transporter",
]


class DataPipeline:
    """Ingests and transforms raw shipment records into weekly route-level aggregations."""

    def __init__(self, filepath: Union[str, Path]):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Shipment records file not found at: {self.filepath}")

    def load_raw_data(self) -> pd.DataFrame:
        """Loads and validates raw shipment CSV records."""
        df = pd.read_csv(self.filepath)
        
        # Verify required columns exist
        missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Shipment records missing required columns: {missing_cols}")
            
        # Parse dates
        df["shipment_date"] = pd.to_datetime(df["shipment_date"])
        
        # Data hygiene checks
        if (df["quantity_tonnes"] <= 0).any():
            raise ValueError("Found non-positive quantity_tonnes in dataset.")
        if (df["distance_km"] <= 0).any():
            raise ValueError("Found non-positive distance_km in dataset.")
        if (df["freight_cost_inr"] < 0).any():
            raise ValueError("Found negative freight_cost_inr in dataset.")
            
        return df

    def process(self) -> pd.DataFrame:
        """
        Executes end-to-end data processing:
        1. Derives route = origin-destination
        2. Assigns week_of as Monday of shipment_date (Monday to Sunday calendar week)
        3. Computes weekly aggregates:
           total_freight_cost = sum(freight_cost_inr)
           total_tonne_km = sum(quantity_tonnes * distance_km)
           cost_per_tonne_km = total_freight_cost / total_tonne_km
        """
        df = self.load_raw_data()
        
        # 1. Derive route
        df["route"] = df["origin"].astype(str) + "-" + df["destination"].astype(str)
        
        # 2. Derive week_of: Monday of each shipment's week
        # Monday is weekday 0, so subtract timedelta(days=weekday)
        df["week_of"] = df["shipment_date"].apply(
            lambda d: d - pd.Timedelta(days=d.weekday())
        )
        
        # 3. Compute tonne-km per individual shipment
        df["shipment_tonne_km"] = df["quantity_tonnes"] * df["distance_km"]
        
        # 4. Group by (route, route_type, week_of)
        # Using unweighted route groupings as specified in case study brief
        grouped = df.groupby(["route", "route_type", "week_of"]).agg(
            total_freight_cost=("freight_cost_inr", "sum"),
            total_tonne_km=("shipment_tonne_km", "sum"),
            shipment_count=("shipment_id", "count"),
            total_quantity_tonnes=("quantity_tonnes", "sum"),
            avg_distance_km=("distance_km", "mean"),
        ).reset_index()
        
        # 5. Core case study formula:
        # Cost per tonne per km = total freight cost / (quantity carried in tonnes x distance travelled in km)
        grouped["cost_per_tonne_km"] = grouped["total_freight_cost"] / grouped["total_tonne_km"]
        
        # Sort chronologically and by route
        grouped = grouped.sort_values(["route", "week_of"]).reset_index(drop=True)
        return grouped


def get_processed_shipment_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """Convenience helper to load and process shipment data."""
    pipeline = DataPipeline(filepath)
    return pipeline.process()
