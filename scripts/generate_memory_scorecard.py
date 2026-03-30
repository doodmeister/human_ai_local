from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

warnings.filterwarnings(
    "ignore",
    message="`clean_up_tokenization_spaces` was not set.*",
    category=FutureWarning,
)

from src.evals.scorecard import generate_memory_quality_scorecard
from src.memory.metrics import metrics_registry


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the deterministic memory/personality quality scorecard.")
    parser.add_argument(
        "--fail-on-gate",
        action="store_true",
        help="Exit with status 1 when any scorecard gate fails.",
    )
    parser.add_argument(
        "--no-telemetry",
        action="store_true",
        help="Do not include the current metrics registry snapshot in the JSON output.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    telemetry_snapshot = None if args.no_telemetry else metrics_registry.export_state()
    scorecard = generate_memory_quality_scorecard(telemetry_snapshot=telemetry_snapshot)
    print(json.dumps(scorecard.to_dict(), indent=2, sort_keys=True))
    if args.fail_on_gate and scorecard.gate_failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())