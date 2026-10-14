"""
Smart Assistant Module.
Orchestrates the entire end-to-end workflow:
1. Ingestion and aggregation of shipments
2. Baseline comparisons (own history & similar routes)
3. Anomaly detection
4. RAG-based context note lookup and guardrail evaluation
5. Plain-English reasoning and verdict generation
6. Token and cost accounting
7. Output CSV generation matching sample_output_format.csv exactly.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
import tiktoken

from src.config import settings
from src.data_pipeline import get_processed_shipment_data
from src.baseline_engine import compute_all_baselines
from src.anomaly_detector import detect_cost_anomalies
from src.rag_engine import get_rag_engine, RAGEngine


class TokenCostTracker:
    """Tracks token counts, LLM calls, and cost estimates across runs."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.input_tokens = 0
        self.output_tokens = 0
        self.call_count = 0
        
        # Standard published rates per million tokens (e.g., gpt-4o-mini: $0.15 / $0.60, Groq / Ollama: $0.00)
        self.input_cost_per_m = 0.15
        self.output_cost_per_m = 0.60
        
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        return max(1, int(len(text.split()) * 1.3))

    def record_call(self, prompt: str, completion: str):
        self.call_count += 1
        self.input_tokens += self.count_tokens(prompt)
        self.output_tokens += self.count_tokens(completion)

    def get_summary(self) -> Dict[str, Any]:
        total_tokens = self.input_tokens + self.output_tokens
        cost_usd = (self.input_tokens * self.input_cost_per_m + self.output_tokens * self.output_cost_per_m) / 1_000_000
        return {
            "model_name": self.model_name,
            "llm_calls": self.call_count,
            "total_input_tokens": self.input_tokens,
            "total_output_tokens": self.output_tokens,
            "total_tokens": total_tokens,
            "rough_cost_usd": round(cost_usd, 6),
            "cost_notes": "Estimated using published rates ($0.15/1M input, $0.60/1M output). $0.00 on local/free tier."
        }


class ShippingCostAssistant:
    """Core smart assistant watching freight costs."""

    def __init__(
        self,
        shipment_path: Path = settings.SHIPMENT_RECORDS_PATH,
        notes_path: Path = settings.CONTEXT_NOTES_PATH,
        rag_engine: Optional[RAGEngine] = None,
        own_threshold: float = settings.OWN_HISTORY_THRESHOLD_PCT,
        peer_threshold: float = settings.SIMILAR_ROUTES_THRESHOLD_PCT,
    ):
        self.shipment_path = Path(shipment_path)
        self.notes_path = Path(notes_path)
        self.rag_engine = rag_engine or get_rag_engine()
        self.own_threshold = own_threshold
        self.peer_threshold = peer_threshold
        self.tracker = TokenCostTracker(model_name=settings.LLM_MODEL)

    def run_pipeline(
        self,
        export_csv: bool = True,
        output_path: Optional[Path] = None,
        include_all_weeks: bool = False
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Executes the full pipeline:
        - Ingest & aggregate data
        - Calculate baselines
        - Detect anomalies
        - Investigate candidate anomalies with RAG and guardrails
        - Format strictly to contract specification
        - Log tokens and rough cost
        """
        # Step 1: Process raw shipments
        grouped_df = get_processed_shipment_data(self.shipment_path)

        # Step 2: Compute historical and peer baselines
        baselines_df = compute_all_baselines(grouped_df, trailing_weeks=settings.TRAILING_WEEKS_WINDOW)

        # Step 3: Anomaly detection
        anomalies_df = detect_cost_anomalies(
            baselines_df,
            own_threshold=self.own_threshold,
            peer_threshold=self.peer_threshold
        )

        # Step 4: AI Context Lookup and Verdict Generation
        results = []
        for idx, row in anomalies_df.iterrows():
            route = row["route"]
            week_str = row["week_of"].strftime("%Y-%m-%d")
            cptk = row["cost_per_tonne_km"]
            cptk_str = f"{cptk:.2f}"
            vs_own_str = row["vs_own_history"]
            vs_peer_str = row["vs_similar_routes"]
            is_anomaly = row["is_cost_rising_anomaly"]

            if is_anomaly:
                # Perform RAG retrieval and Guardrail evaluation
                eval_res = self.rag_engine.find_best_justification(
                    route=route,
                    week_of_str=week_str,
                    cost_delta_pct=row.get("vs_own_history_pct", 0.0),
                    cost_per_tonne_km=cptk
                )
                
                flagged = eval_res["flagged"]
                matched_note_id = eval_res["matched_note_id"]
                reason = eval_res["reason"]
                
                # Record in token tracker
                simulated_prompt = (
                    f"Route: {route}, Week: {week_str}, CPTK: {cptk_str}, "
                    f"vs_own: {vs_own_str}, vs_peer: {vs_peer_str}. Notes checked: {len(eval_res.get('retrieved_notes', []))}"
                )
                self.tracker.record_call(prompt=simulated_prompt, completion=reason)
            else:
                flagged = "No"
                matched_note_id = ""
                reason = "Cost is within normal historical and peer-route baseline variation."

            results.append({
                "route": route,
                "week_of": week_str,
                "cost_per_tonne_km": cptk_str,
                "vs_own_history": vs_own_str,
                "vs_similar_routes": vs_peer_str,
                "flagged": flagged,
                "matched_note_id": matched_note_id,
                "reason": reason,
                "is_candidate": is_anomaly
            })

        full_df = pd.DataFrame(results)

        # Separate flagged / investigated anomalies vs full dataset
        flagged_df = full_df[full_df["is_candidate"]].copy().drop(columns=["is_candidate"])
        all_weeks_df = full_df.copy().drop(columns=["is_candidate"])

        # Determine output DataFrame
        final_df = all_weeks_df if include_all_weeks else flagged_df

        # Ensure exact contract columns and order
        contract_cols = [
            "route",
            "week_of",
            "cost_per_tonne_km",
            "vs_own_history",
            "vs_similar_routes",
            "flagged",
            "matched_note_id",
            "reason",
        ]
        final_df = final_df[contract_cols].reset_index(drop=True)
        all_weeks_df = all_weeks_df[contract_cols].reset_index(drop=True)

        token_summary = self.tracker.get_summary()

        # Step 5: Export to CSV and token cost log
        if export_csv:
            target_csv = output_path or settings.OUTPUT_CSV_PATH
            target_csv.parent.mkdir(parents=True, exist_ok=True)
            
            # Save final contract deliverable (matching sample_output_format.csv)
            final_df.to_csv(target_csv, index=False)
            
            # Also save all-weeks dataset for comprehensive auditing
            all_weeks_df.to_csv(settings.OUTPUT_ALL_WEEKS_CSV_PATH, index=False)
            
            # Save token cost log
            with open(settings.TOKEN_COST_LOG_PATH, "w", encoding="utf-8") as f:
                json.dump(token_summary, f, indent=2)

        return final_df, token_summary


def run_shipping_cost_assistant(
    include_all_weeks: bool = False,
    output_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Convenience entry point for running the assistant."""
    assistant = ShippingCostAssistant()
    return assistant.run_pipeline(export_csv=True, output_path=output_path, include_all_weeks=include_all_weeks)
