import pandas as pd
import re
import spacy

nlp = spacy.load("en_core_web_sm")

def clean_body(body: str) -> str:
    """Cleans body of email. Removes stop words and punctuation. (Add more).
    
    Args:
        body: String containing body of email
        
    Returns:
        String containing cleaned body
    """

    #add some more cleaning here

    #detect standard signatures which don't add too much context to tone of email
    #reg expressions here capture the signature and everything afterwards (name, maybe contact info, etc.)
    standard_signatures_regex = [
        r"\nBest,.*",
        r"\nBest regards,.*",
        r"\nRegards,.*",
        r"\nThanks in advance,.*",
        r"\nMany thanks,.*",
        r"\nSincerely,.*",
        r"\nCheers,.*",
        r"\nKind regards,.*",
        r'\n\w+ \| .*\n' # for business sign-offs, where people do Name | Company or something
    ]
    
    #look for signature - if find, remove it and everything afterwards
    for regex in standard_signatures_regex:
        text = re.sub(regex,"",text,flags=re.IGNORECASE)    

    #create a list of token objects with relevant attributes
    processed_body = nlp(body) 

    #put the tokens back together into a cleaned string, which excludes stop words and punctuation
    cleaned_body = " ".join([token.lemma_ for token in processed_body if not (token.is_stop or token.is_punct)])

    return cleaned_body

def clean_subject(subj: str) -> str:
    """Cleans subject line of email.
    
    Args:
        body: String containing subject line of email
        
    Returns:
        String containing cleaned subject line
    """
    #simpler cleaning for subject line to avoid excessively removing relevant context

    subj = subj.lower()

    #use regex to remove possible "re:" or "fwd:" prefixes
    #may or may not want to do this; useful context or adds noise?
    subj = re.sub(r"^(re|fwd):","",subj) 

    #removes leading and trailing whitespace
    return subj.strip()

def clean_email(email: pd.DataFrame) -> pd.DataFrame:
    """Cleans body and subject line of email.
    
    Args:
        email: DataFrame object representing an email
    """
    email["subject"] = clean_subject(email["subject"])
    email["body"] = clean_body(email["body"])




