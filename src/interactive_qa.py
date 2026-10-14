"""
Interactive Q&A Module (Stretch Goal).
Allows users to ask plain-English questions about the assistant's findings
(e.g., "Why did Chennai-Bangalore get pricier in March 2025?",
       "What routes spiked in November 2024?",
       "Why did Ahmedabad-Mumbai have a surcharge in January?")
and answers accurately using the assistant's own computed results and context notes.
"""

import re
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime

from src.config import settings


class ShippingCostQAAgent:
    """Natural language question answering assistant for shipping cost analysis."""

    MONTHS = {
        "january": 1, "jan": 1,
        "february": 2, "feb": 2,
        "march": 3, "mar": 3,
        "april": 4, "apr": 4,
        "may": 5,
        "june": 6, "jun": 6,
        "july": 7, "jul": 7,
        "august": 8, "aug": 8,
        "september": 9, "sep": 9, "sept": 9,
        "october": 10, "oct": 10,
        "november": 11, "nov": 11,
        "december": 12, "dec": 12,
    }

    def __init__(self, output_df: Optional[pd.DataFrame] = None, all_weeks_df: Optional[pd.DataFrame] = None):
        if output_df is not None:
            self.output_df = output_df
        elif settings.OUTPUT_CSV_PATH.exists():
            self.output_df = pd.read_csv(settings.OUTPUT_CSV_PATH)
        else:
            self.output_df = pd.DataFrame()

        if all_weeks_df is not None:
            self.all_weeks_df = all_weeks_df
        elif settings.OUTPUT_ALL_WEEKS_CSV_PATH.exists():
            self.all_weeks_df = pd.read_csv(settings.OUTPUT_ALL_WEEKS_CSV_PATH)
        else:
            self.all_weeks_df = self.output_df

        if settings.CONTEXT_NOTES_PATH.exists():
            self.notes_df = pd.read_csv(settings.CONTEXT_NOTES_PATH)
        else:
            self.notes_df = pd.DataFrame()

    def parse_query_entities(self, query: str) -> Dict[str, Any]:
        """Extracts route, month, and year from natural language query."""
        q_lower = query.lower()
        
        # 1. Detect route
        matched_route = None
        routes = [
            "delhi-jaipur", "ahmedabad-mumbai", "mumbai-pune",
            "chennai-bangalore", "kolkata-bhubaneswar", "mumbai-delhi", "delhi-chennai"
        ]
        for r in routes:
            parts = r.split("-")
            if r in q_lower or (parts[0] in q_lower and parts[1] in q_lower):
                # Format to title case
                matched_route = "-".join([p.capitalize() for p in parts])
                break
                
        # 2. Detect month
        matched_month = None
        for m_name, m_num in self.MONTHS.items():
            if re.search(r'\b' + re.escape(m_name) + r'\b', q_lower):
                matched_month = m_num
                break
                
        # 3. Detect year
        matched_year = None
        year_match = re.search(r'\b(202[4-6])\b', q_lower)
        if year_match:
            matched_year = int(year_match.group(1))
            
        return {
            "route": matched_route,
            "month": matched_month,
            "year": matched_year,
            "raw_query": query
        }

    def answer(self, query: str) -> str:
        """Answers a user's question using computed shipping data and context notes."""
        if self.all_weeks_df.empty:
            return "No analysis data is currently loaded. Please run the assistant pipeline first."

        entities = self.parse_query_entities(query)
        route = entities["route"]
        month = entities["month"]
        year = entities["year"]

        df = self.all_weeks_df.copy()
        df["dt"] = pd.to_datetime(df["week_of"])

        # Filter by route if present
        if route:
            df = df[df["route"].str.lower() == route.lower()]

        # Filter by year if present
        if year:
            df = df[df["dt"].dt.year == year]

        # Filter by month if present
        if month:
            df = df[df["dt"].dt.month == month]

        if df.empty:
            scope = f"for {route}" if route else ""
            time_scope = f"in {datetime(year or 2024, month or 1, 1).strftime('%B')} {year or ''}".strip() if month or year else ""
            return f"I couldn't find any shipment records matching your criteria {scope} {time_scope}."

        # Separate flagged / investigated rows vs normal rows
        flagged_rows = df[df["flagged"].isin(["Yes", "No (justified)"])]

        # Specific query about a route and time
        if route and (month or year):
            month_name = datetime(2024, month, 1).strftime("%B") if month else ""
            time_str = f"{month_name} {year or ''}".strip()

            if not flagged_rows.empty:
                responses = []
                for _, r in flagged_rows.iterrows():
                    w = r["week_of"]
                    cptk = r["cost_per_tonne_km"]
                    vs_own = r["vs_own_history"]
                    vs_peer = r["vs_similar_routes"]
                    verdict = r["flagged"]
                    note_id = r["matched_note_id"] if pd.notna(r["matched_note_id"]) else ""
                    reason = r["reason"]

                    verdict_desc = "justified by operational records" if verdict == "No (justified)" else "unexplained and flagged for review"
                    
                    resp = (
                        f"In the week of **{w}**, **{route}** saw a cost increase to **INR {cptk} per tonne-km** "
                        f"({vs_own}, {vs_peer}).\n\n"
                        f"- **Verdict:** {verdict} ({verdict_desc})\n"
                    )
                    if note_id:
                        resp += f"- **Matched Note:** {note_id}\n"
                    resp += f"- **Explanation:** {reason}"
                    responses.append(resp)
                return "\n\n---\n\n".join(responses)
            else:
                avg_cost = df["cost_per_tonne_km"].astype(float).mean()
                return (
                    f"During **{time_str}**, shipping costs on **{route}** remained stable and within normal baseline variation, "
                    f"averaging **INR {avg_cost:.2f} per tonne-km**. No anomalous spikes or disruptions were flagged."
                )

        # General inquiry: "What routes were flagged?" or "What spikes occurred?"
        if not flagged_rows.empty:
            summary = [
                f"I found **{len(flagged_rows)}** cost anomaly events in the requested scope:"
            ]
            for _, r in flagged_rows.head(10).iterrows():
                summary.append(
                    f"• **{r['route']}** (week of {r['week_of']}): INR {r['cost_per_tonne_km']}/tonne-km "
                    f"({r['vs_own_history']}) — **{r['flagged']}**"
                    + (f" [Note {r['matched_note_id']}]" if pd.notna(r['matched_note_id']) and r['matched_note_id'] else "")
                )
            if len(flagged_rows) > 10:
                summary.append(f"...and {len(flagged_rows) - 10} more events.")
            return "\n".join(summary)

        return (
            "All evaluated routes in this period operated within normal historical and peer baselines. "
            "No ungrounded or unexplained spikes were detected."
        )


def ask_assistant(question: str) -> str:
    """Convenience helper to ask a question to the assistant."""
    agent = ShippingCostQAAgent()
    return agent.answer(question)
