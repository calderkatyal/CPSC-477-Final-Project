import os
import zipfile
import kaggle

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
RAW_DIR = os.path.join(BASE_DIR, "raw")
ZIP_PATH = os.path.join(BASE_DIR, "hillary-emails.zip")
DATASET = "kaggle/hillary-clinton-emails"

NEEDED_FILES = {
    "Aliases.csv",
    "EmailReceivers.csv",
    "Emails.csv",
    "Persons.csv"
}

os.makedirs(RAW_DIR, exist_ok=True)

print("📦 Downloading dataset...")
kaggle.api.dataset_download_files(DATASET, path=BASE_DIR, unzip=False)

print("📂 Extracting files...")
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    for file in zip_ref.namelist():
        filename = os.path.basename(file)
        if filename in NEEDED_FILES:
            print(f"  ✅ Extracting: {filename}")
            zip_ref.extract(file, RAW_DIR)

print("🧹 Cleaning up zip file...")
os.remove(ZIP_PATH)

print(f"Done! Files are in: {RAW_DIR}")
