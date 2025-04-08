import os
import zipfile
import kaggle

# Dataset and files
DATASET = "kaggle/hillary-clinton-emails"
FILES = [
    "Aliases.csv",
    "EmailReceivers.csv",
    "Emails.csv",  # <- this one will come zipped
    "Persons.csv"
]

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "raw"))
os.makedirs(BASE_DIR, exist_ok=True)

print("📥 Downloading and extracting Hillary Clinton email dataset...")

for file in FILES:
    file_path = os.path.join(BASE_DIR, file)
    print(f"  ⬇️ Downloading: {file}")

    # Download the file
    kaggle.api.dataset_download_file(
        DATASET,
        file_name=file,
        path=BASE_DIR,
        force=True
    )

    # Only Emails.csv comes down as a ZIP (Emails.csv.zip)
    if file == "Emails.csv":
        zip_path = os.path.join(BASE_DIR, f"{file}.zip")

        # Rename the downloaded file to match the real ZIP filename
        os.rename(file_path, zip_path)

        # Extract it
        print(f"  📂 Extracting: {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(BASE_DIR)
        os.remove(zip_path)
        print("  ✅ Extracted Emails.csv")
    else:
        print(f"  ✅ Saved: {file}")

print(f"\n✅ All files ready and UTF-8 compatible in: {BASE_DIR}")
