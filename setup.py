import os
import kaggle

# Define dataset and target files
DATASET = "kaggle/hillary-clinton-emails"
FILES = [
    "Aliases.csv",
    "EmailReceivers.csv",
    "Emails.csv",
    "Persons.csv"
]

# Define destination folder
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "raw"))
os.makedirs(BASE_DIR, exist_ok=True)

# Download files
print("Downloading files from Kaggle...")
for file in FILES:
    print(f"  ⬇️ {file}")
    kaggle.api.dataset_download_file(DATASET, file_name=file, path=BASE_DIR, force=True)

for file in FILES:
    zip_path = os.path.join(BASE_DIR, f"{file}.zip")
    csv_path = os.path.join(BASE_DIR, file)
    if os.path.exists(zip_path):
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(BASE_DIR)
        os.remove(zip_path)
        print(f"Extracted: {file}")

print(f"\nAll files downloaded to: {BASE_DIR}")
