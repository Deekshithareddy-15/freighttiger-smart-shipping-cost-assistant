import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.eval_harness import run_eval_harness


def main():
    results = run_eval_harness()
    if results["math_accuracy_pct"] == 100.0 and results["hallucination_rate_pct"] == 0.0:
        print("\n>>> ALL EVALUATION CHECKS PASSED WITH FLYING COLORS! <<<")
    else:
        print("\n>>> EVALUATION REPORTED DEFECTS! <<<")


if __name__ == "__main__":
    main()
