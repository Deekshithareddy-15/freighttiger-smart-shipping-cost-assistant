import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.assistant import run_shipping_cost_assistant
from src.config import settings


def main():
    print("=" * 65)
    print("FREIGHTTIGER SMART ASSISTANT: RUNNING FULL PIPELINE")
    print("=" * 65)
    
    # Run full assistant pipeline
    df, summary = run_shipping_cost_assistant()
    
    print("\n[OK] Pipeline completed successfully.")
    print(f"[+] Output rows: {len(df)}")
    print(f"[+] Output CSV path: {settings.OUTPUT_CSV_PATH}")
    print(f"[+] All-weeks dataset path: {settings.OUTPUT_ALL_WEEKS_CSV_PATH}")
    print(f"[+] Token cost log path: {settings.TOKEN_COST_LOG_PATH}")
    
    print("\n--- Output Preview (first 10 rows) ---")
    print(df.head(10).to_string())
    
    print("\n--- Verdict Counts ---")
    for verdict, count in df["flagged"].value_counts().items():
        print(f"  • {verdict}: {count}")
        
    print("\n--- Token & Cost Accounting ---")
    for k, v in summary.items():
        print(f"  • {k}: {v}")
    print("=" * 65)


if __name__ == "__main__":
    main()
