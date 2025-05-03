import os
import zipfile
import kaggle
from src.config import RAW_DIR

# Dataset and files
DATASET = "kaggle/hillary-clinton-emails"
FILES = [
    "Aliases.csv",
    "EmailReceivers.csv",
    "Emails.csv",  
    "Persons.csv"
]
os.makedirs(RAW_DIR, exist_ok=True)

print("📥 Downloading and extracting Hillary Clinton email dataset...")

for file in FILES:
    file_path = os.path.join(RAW_DIR, file)
    print(f"  ⬇️ Downloading: {file}")

    # Download the file
    kaggle.api.dataset_download_file(
        DATASET,
        file_name=file,
        path=RAW_DIR,
        force=True
    )

   
    if file == "Emails.csv":
        zip_path = os.path.join(RAW_DIR, f"{file}.zip")

     
        os.rename(file_path, zip_path)

        print(f"  📂 Extracting: {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(RAW_DIR)
        os.remove(zip_path)
        print("  ✅ Extracted Emails.csv")
    else:
        print(f"  ✅ Saved: {file}")

print(f"\n✅ All files ready and UTF-8 compatible in: {RAW_DIR}")
