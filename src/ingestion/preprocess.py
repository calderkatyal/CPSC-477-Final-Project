"""
Email preprocessing utilities.
"""
from dataloader import load, save
import re
import pandas as pd
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_CSV_PATH = os.path.join(BASE_DIR, "data", "raw", "emails.csv")
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed", "emails.parquet")


def parse_headers(txt: str) -> dict:
    """Parse email headers and body from raw text.
    
    Args:
        txt: Raw email text
        
    Returns:
        Dictionary containing parsed email components
    """
    flags = re.MULTILINE
    s = re.search(r"^From:\s*(.*)$", txt, flags)
    r = re.search(r"^To:\s*(.*)$", txt, flags)
    c = re.search(r"^Cc:\s*(.*)$", txt, flags)
    sbj = re.search(r"^Subject:\s*(.*)$", txt, flags)
    parts = txt.split("\n\n", 1)
    b = parts[1].strip() if len(parts) > 1 else txt.strip()
    #The way our tool would be applied is that a user would use it to search their own email. 
    #As such, the query "about my weekly session" has a very different meaning than "about John's weekly session".
    #Moreover, as mentioned, we may want go beyond the most basic method of finding relevant emails
    #which is basically just using a tool to generate embeddings for queries and emails.
    #(For instance, one thing we discussed was incorporating the information of the person you are corresponding with
    #into the embedding for a given email, because that adds significant context. This embedding would be generated
    #using all of the emails between you and this person (probably with an RNN/LSTM or something like that)).
    #For these reasons, it might be worth
    #it to organize our emails in our database accordingly. Currently, our dataset is a bunch of emails. We might want 
    #to reorganize. We could have that, for every person (email address) in the database, our dataset has their "email box",
    #meaning a "sent" group and a "received" group. 
    #For illustration, an example path in this dataset would be: Dataset -> John Smith -> Sent -> An Email.

    return {
        "sender": s.group(1) if s else "Unknown",
        "recipient": r.group(1) if r else "Unknown",
        "cc": c.group(1) if c else "Unknown",
        "subject": sbj.group(1) if sbj else "No Subject",
        "body": b
    }

def preprocess_emails(input_path: str, output_path: str) -> pd.DataFrame:
    """Preprocess emails from raw CSV to structured format.
    
    Args:
        input_path: Path to raw CSV file
        output_path: Path to save processed data
        
    Returns:
        DataFrame containing processed emails
    """
    df = load(input_path)
    df_parsed = df["message"].apply(parse_headers).apply(pd.Series)
    df = df.join(df_parsed)  
    save(df, output_path)


if __name__ == "__main__":
    preprocess_emails(RAW_CSV_PATH, PROCESSED_PATH)
