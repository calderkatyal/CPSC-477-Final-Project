from ingestion.dataloader import load, save

def preprocess_emails(input_path, output_path):
    print("Loading dataset...")
    df = load(input_path)
    print("Saving preprocessed emails...")
    save(df, output_path)

if __name__ == "__main__":
    RAW_CSV_PATH = "data/raw/enron_emails.csv"
    PROCESSED_PATH = "data/processed/cleaned_emails.parquet"

    preprocess_emails(RAW_CSV_PATH, PROCESSED_PATH)