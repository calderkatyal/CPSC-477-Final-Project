from src.ingestion.dataloader import load, save
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_CSV_PATH = os.path.join(BASE_DIR, "data", "raw", "emails.csv")
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed", "emails.parquet")

def preprocess_emails(input_path, output_path):
    df = load(input_path)
    save(df, output_path)

if __name__ == "__main__":
    preprocess_emails(RAW_CSV_PATH, PROCESSED_PATH)
