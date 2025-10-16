import os
from eda import run_all

if __name__ == "__main__":
    here = os.path.dirname(__file__)
    run_all(here)
    print("Done. See figures/, eda_summary.json, and quick_metrics.txt.")
