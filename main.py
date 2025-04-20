import os
import subprocess
from src.config import DATA_DIR, PROCESSED_DIR
from src.hybrid_search.hybrid_search import run_search_interface

if __name__ == "__main__":
    if not os.path.isdir(DATA_DIR):
        subprocess.run(["python", "-m", "src.scripts.download_data"], check=True)
    if not os.path.isdir(PROCESSED_DIR):
        subprocess.run(["python", "-m", "src.preprocessing.preprocess"], check=True)

    run_search_interface()
