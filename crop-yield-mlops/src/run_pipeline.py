"""One command for the whole M1 pipeline: prepare data -> train -> test.

    python -m src.run_pipeline
"""

import subprocess
import sys

from src.prepare_data import prepare_data
from src.train import train


def main():
    prepare_data()
    summary = train()
    code = subprocess.call([sys.executable, "-m", "pytest", "tests/", "-q"])
    if code != 0:
        sys.exit(code)
    if not summary["beats_baseline"]:
        print("WARNING: best model does not beat the last-year baseline")
        sys.exit(1)
    print("Pipeline OK: data prepared, model trained and registered, tests pass.")


if __name__ == "__main__":
    main()
