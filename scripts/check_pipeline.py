"""Exercise the full lineup path against tracked models/data in an isolated copy.

The network ingest has its own source check. This smoke check supplies cached
data and simulates an actionable deadline; preparation, projections, CBC, chip
revision, rendering and persistence all run normally.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="fpl-pipeline-check-") as tmp:
        root = Path(tmp)
        shutil.copytree(repository / "src", root / "src",
                        ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(repository / "data", root / "data",
                        ignore=shutil.ignore_patterns(".fpl_ci_cache"))
        check = root / "check.py"
        check.write_text('''from pathlib import Path
import sys
from unittest.mock import patch
import pandas as pd
root = Path(__file__).parent
sys.path.insert(0, str(root / "src"))
import main
# Exercise generation regardless of the calendar. Production skip behavior is
# covered separately; none of the model/solver/persistence logic is replaced.
with patch.object(main, "refresh_data"), \
     patch.object(main, "_gw_in_play", return_value=False), \
     patch.object(main, "_season_complete", return_value=False), \
     patch.object(main, "_last_completed_gw", return_value=0):
    main.main()
outcome = (root / "outcome.txt").read_text()
snapshot = pd.read_csv(root / "data/processed/squad_snapshot.csv")
assert snapshot["id"].nunique() == 15
assert snapshot["in_xi"].sum() == 11
assert snapshot["is_captain"].sum() == 1
assert "generated=true" in outcome
assert (root / "data/processed/lineup.md").read_text(encoding="utf-8")
print("PASS: model preparation, projections, optimizer, chips and 15-player lineup")
''', encoding="utf-8")
        env = dict(os.environ, GITHUB_OUTPUT=str(root / "outcome.txt"),
                   GITHUB_STEP_SUMMARY=str(root / "summary.md"))
        subprocess.run([sys.executable, "-u", str(check)], env=env, check=True,
                       timeout=600)


if __name__ == "__main__":
    main()
