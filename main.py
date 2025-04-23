import os
import subprocess
import argparse
from src.config import DATA_DIR, PROCESSED_DIR
from src.hybrid_search.hybrid_search import run_search_interface

if __name__ == "__main__":
    if not os.path.isdir(DATA_DIR):
        subprocess.run(["python", "-m", "src.scripts.download_data"], check=True)
    if not os.path.isdir(PROCESSED_DIR):
        subprocess.run(["python", "-m", "src.preprocessing.preprocess"], check=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--is_test", action="store_true", help="Run in test mode")
    args = parser.parse_args()
    run_search_interface(args.is_test)
