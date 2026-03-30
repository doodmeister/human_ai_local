from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evals.scorecard import generate_memory_quality_scorecard
from src.memory.metrics import metrics_registry


def main() -> int:
    scorecard = generate_memory_quality_scorecard(telemetry_snapshot=metrics_registry.export_state())
    print(json.dumps(scorecard.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())