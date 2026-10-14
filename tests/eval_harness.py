"""
Evaluation Harness Module.
Provides automated benchmark scoring against a labeled evaluation dataset:
- Baseline Math Accuracy (100% target)
- Verdict Accuracy (Yes / No (justified))
- Note Citation Accuracy (N001, N002, etc.)
- Hallucination Rate (Strict 0.0% target on negative traps N004, N005, N006, N007, N008, N009, N010)
- Contract Schema Compliance
"""

import sys
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from src.assistant import ShippingCostAssistant
from src.config import settings


# Benchmark labeled cases derived strictly from case study specifications and verified data
LABELED_BENCHMARK = [
    {
        "case_name": "Sample Case 1: Unexplained spike (Delhi-Jaipur 2024-11-11)",
        "route": "Delhi-Jaipur",
        "week_of": "2024-11-11",
        "expected_cptk": "4.17",
        "expected_own_history": "+35.5% vs this route's past average",
        "expected_similar_routes": "+21.0% vs similar-length routes this week",
        "expected_flagged": "Yes",
        "expected_note_id": "",
        "notes": "Cost spike with no matching note. Must be marked unexplained."
    },
    {
        "case_name": "Sample Case 2: Justified festival surcharge (Ahmedabad-Mumbai 2025-01-20)",
        "route": "Ahmedabad-Mumbai",
        "week_of": "2025-01-20",
        "expected_cptk": "3.29",
        "expected_own_history": "+29.5% vs this route's past average",
        "expected_similar_routes": "+22.5% vs similar-length routes this week",
        "expected_flagged": "No (justified)",
        "expected_note_id": "N002",
        "notes": "Festival week surcharge. Must cite N002 and be marked No (justified)."
    },
    {
        "case_name": "Sample Case 3: Trap rejection / stable demand note (Mumbai-Pune 2025-09-15)",
        "route": "Mumbai-Pune",
        "week_of": "2025-09-15",
        "expected_cptk": "3.98",
        "expected_own_history": "+9.2% vs this route's past average",
        "expected_similar_routes": "+23.6% vs similar-length routes this week",
        "expected_flagged": "Yes",
        "expected_note_id": "",
        "notes": "Closest note N006 states stable demand. Must NOT cite N006 as justification."
    },
    {
        "case_name": "Case 4: Flood disruption onset (Chennai-Bangalore 2025-02-24)",
        "route": "Chennai-Bangalore",
        "week_of": "2025-02-24",
        "expected_cptk": "3.58",
        "expected_own_history": "+31.5% vs this route's past average",
        "expected_similar_routes": "+38.7% vs similar-length routes this week",
        "expected_flagged": "No (justified)",
        "expected_note_id": "N001",
        "notes": "Highway flooding disruption window. Must cite N001."
    },
    {
        "case_name": "Case 5: Flood disruption continuation (Chennai-Bangalore 2025-03-03)",
        "route": "Chennai-Bangalore",
        "week_of": "2025-03-03",
        "expected_cptk": "3.48",
        "expected_own_history": "+22.8% vs this route's past average",
        "expected_similar_routes": "+32.0% vs similar-length routes this week",
        "expected_flagged": "No (justified)",
        "expected_note_id": "N001",
        "notes": "Second week of flood disruption. Must cite N001."
    },
    {
        "case_name": "Case 6: Post-flood repair completion (Chennai-Bangalore 2025-03-17)",
        "route": "Chennai-Bangalore",
        "week_of": "2025-03-17",
        "expected_cptk": "3.47",
        "expected_own_history": "+13.7% vs this route's past average",
        "expected_similar_routes": "+33.5% vs similar-length routes this week",
        "expected_flagged": "Yes",
        "expected_note_id": "",
        "notes": "Note N009 reports recovery to normal conditions. Must NOT be marked justified."
    }
]

# Negative trap notes that must NEVER be cited as justification
TRAP_NOTE_IDS = {"N004", "N005", "N006", "N007", "N008", "N009", "N010"}


class EvaluationHarness:
    """Executes automated tests and computes accuracy, precision, and hallucination scores."""

    def __init__(self):
        self.assistant = ShippingCostAssistant()

    def run_eval(self) -> Dict[str, Any]:
        print("Running full pipeline for evaluation...")
        df, summary = self.assistant.run_pipeline(export_csv=False)

        total_cases = len(LABELED_BENCHMARK)
        math_matches = 0
        verdict_matches = 0
        note_matches = 0
        case_results = []

        for case in LABELED_BENCHMARK:
            route = case["route"]
            week = case["week_of"]
            match = df[(df["route"] == route) & (df["week_of"] == week)]

            if match.empty:
                case_results.append({
                    "case": case["case_name"],
                    "passed": False,
                    "error": f"Route-week {route} on {week} not present in output."
                })
                continue

            row = match.iloc[0]
            cptk_ok = (row["cost_per_tonne_km"] == case["expected_cptk"])
            own_ok = (row["vs_own_history"] == case["expected_own_history"])
            peer_ok = (row["vs_similar_routes"] == case["expected_similar_routes"])
            math_passed = cptk_ok and own_ok and peer_ok
            if math_passed:
                math_matches += 1

            verdict_ok = (row["flagged"] == case["expected_flagged"])
            if verdict_ok:
                verdict_matches += 1

            note_ok = (str(row["matched_note_id"]).strip() == str(case["expected_note_id"]).strip())
            if note_ok:
                note_matches += 1

            passed = math_passed and verdict_ok and note_ok
            case_results.append({
                "case": case["case_name"],
                "passed": passed,
                "math_ok": math_passed,
                "verdict_ok": verdict_ok,
                "note_ok": note_ok,
                "actual_verdict": row["flagged"],
                "actual_note": row["matched_note_id"],
                "reason": row["reason"]
            })

        # Check for any hallucinations across the ENTIRE output DataFrame
        all_justified_rows = df[df["flagged"] == "No (justified)"]
        hallucinated_rows = all_justified_rows[all_justified_rows["matched_note_id"].isin(TRAP_NOTE_IDS)]
        total_hallucinations = len(hallucinated_rows)

        # Output contract schema checks
        required_columns = [
            "route", "week_of", "cost_per_tonne_km", "vs_own_history",
            "vs_similar_routes", "flagged", "matched_note_id", "reason"
        ]
        schema_compliant = (df.columns.tolist() == required_columns)

        results = {
            "total_benchmark_cases": total_cases,
            "math_accuracy_pct": round((math_matches / total_cases) * 100.0, 2),
            "verdict_accuracy_pct": round((verdict_matches / total_cases) * 100.0, 2),
            "note_citation_accuracy_pct": round((note_matches / total_cases) * 100.0, 2),
            "hallucination_rate_pct": round((total_hallucinations / max(1, len(all_justified_rows))) * 100.0, 2),
            "trap_notes_avoided": total_hallucinations == 0,
            "schema_contract_compliant": schema_compliant,
            "total_evaluated_anomalies": len(df),
            "cases": case_results
        }
        return results

    def print_report(self, results: Dict[str, Any]):
        print("\n" + "=" * 65)
        print("FREIGHTTIGER SMART ASSISTANT: EVALUATION HARNESS REPORT")
        print("=" * 65)
        print(f"Total Labeled Benchmark Cases : {results['total_benchmark_cases']}")
        print(f"Core Baseline Math Accuracy   : {results['math_accuracy_pct']}%")
        print(f"Verdict Classification Rate   : {results['verdict_accuracy_pct']}%")
        print(f"Exact Note Citation Accuracy  : {results['note_citation_accuracy_pct']}%")
        print(f"Hallucination Rate            : {results['hallucination_rate_pct']}% (0.0% expected)")
        print(f"Trap Notes Avoided (N004-N010): {'PASSED' if results['trap_notes_avoided'] else 'FAILED'}")
        print(f"Schema Contract Compliance    : {'PASSED' if results['schema_contract_compliant'] else 'FAILED'}")
        print("-" * 65)
        print("Individual Case Results:")
        for c in results["cases"]:
            status = "[PASS]" if c["passed"] else "[FAIL]"
            print(f"  {status} {c['case']}")
            if not c["passed"]:
                print(f"         Actual: verdict={c.get('actual_verdict')}, note={c.get('actual_note')}")
        print("=" * 65)


def run_eval_harness() -> Dict[str, Any]:
    harness = EvaluationHarness()
    results = harness.run_eval()
    harness.print_report(results)
    return results


if __name__ == "__main__":
    res = run_eval_harness()
    if res["math_accuracy_pct"] < 100 or res["hallucination_rate_pct"] > 0:
        sys.exit(1)
    sys.exit(0)
