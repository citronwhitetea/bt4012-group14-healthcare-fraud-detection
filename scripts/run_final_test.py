"""Display saved final results by default; explicit evaluation is globally guarded.

Examples (from any directory):
  python scripts/run_final_test.py
  python scripts/run_final_test.py --evaluate --run artifacts/boosting/RUN_ID

The second command refuses after any shared evaluation has started/completed.
The exact original execution helper and test cell are archived under provenance/.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.final_evaluation import load_saved_results, run_selected_test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluate', action='store_true', help='Request the one permitted final evaluation.')
    parser.add_argument('--run', help='Repository-relative run folder, required with --evaluate.')
    args = parser.parse_args()
    if args.evaluate:
        if not args.run:
            parser.error('--run is required with --evaluate')
        print(run_selected_test(ROOT, ROOT / args.run))
    else:
        saved = load_saved_results(ROOT)
        print('Saved team result (read only):', saved['selection']['selected_run'])
        print(json.dumps(saved['test'], indent=2))


if __name__ == '__main__':
    main()
