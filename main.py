import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 

import subprocess
import argparse
from src.config import DATA_DIR, PROCESSED_DIR
from src.hybrid_search.hybrid_search import run_search_interface
from src.utils import set_global_seed 

if __name__ == "__main__":
    if not os.path.isdir(DATA_DIR):
        subprocess.run(["python", "-m", "src.scripts.download_data"], check=True)
    if not os.path.isdir(PROCESSED_DIR):
        subprocess.run(["python", "-m", "src.preprocessing.preprocess"], check=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--is_test", action="store_true", help="Run in test mode")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        print(f"📌 Setting random seed to {args.seed}")
        set_global_seed(args.seed)

    run_search_interface(args.is_test, seed=args.seed)
