import re

from sqlalchemy import create_engine, text

# Database connection
DB_URL = "postgresql://postgres:password@localhost:5432/emails_db"
engine = create_engine(DB_URL)

def split_email_chain(email_chain: str):
    #Emails in chain are separated by email metadata
    email_separator = r'(?=From: .*\nSent: .*\nTo: .*\nSubject: .*\n)'

    #Split chain into separate emails using the separator
    email_texts = re.split(email_separator, email_chain)

    #Lists for email bodies and metadata
    email_bodies = []
    email_metadata = []

    #Regex patterns for extracting metadata for each email in chain
    metadata_pattern = re.compile(
        r'From: (.*?)\nSent: (.*?)\nTo: (.*?)\nSubject: (.*?)\n',
        re.DOTALL
    )

    #Parse chain into email_bodies and email_metadata lists
    for email in email_texts:
        if email.strip(): #Make sure email isn't empty
            #Get metadata
            metadata_match = metadata_pattern.match(email)

            if metadata_match:
                #Get metadata
                from_text, sent_text, to_text, subject_text = metadata_match.groups()
                #Body comes after metadata
                body_text = email[metadata_match.end():].strip()

                #Add metadata and email body to the lists
                email_metadata.append({
                    'from': from_text,
                    'date': sent_text,
                    'to': to_text,
                    'subject': subject_text
                })
                email_bodies.append(body_text)

    return email_bodies, email_metadata

def format_chain_as_conversation(metadata_list, email_bodies):

    #Get date and subject of latest email in chain, to serve as header for conversation
    latest_email = metadata_list[0]
    latest_email_header = f"{latest_email['subject']} ({latest_email['date']})\n"

    #Reverse the lists so emails are in order from earliest to latest
    reversed_metadata = metadata_list[::-1]
    reversed_bodies = email_bodies[::-1]

    #Create string from rest of emails to be like conversation
    #Looks like:
    #Jill: I want to build a tree house.
    #Jack: Yes, that sounds like a great idea.
    formatted_string = "\n".join(
        f"{metadata['from']}: {body}" for metadata, body in zip(reversed_metadata, reversed_bodies)
    )

    #Combine the header and the conversation string
    return latest_email_header + formatted_string

#Helper to delete email that has matching sender, recipient, date, subject
def delete_email(sender, recipient, date_sent, subject):
    with engine.begin() as conn:  
        delete_query = text("""
            DELETE FROM emails 
            WHERE "ExtractedFrom" = :sender 
            AND "ExtractedTo" = :recipient 
            AND "ExtractedDateSent" = :date_sent 
            AND "ExtractedSubject" = :subject;
        """)
       
        result = conn.execute(delete_query, {
            "sender": sender,
            "recipient": recipient,
            "date_sent": date_sent,
            "subject": subject
        })


#For each chain of emails (A, B, C), the database also has an entry which is the chain (B,C) along with the entry for the original email, C. 
#We should remove the (B,C) and C entries.
#This function takes in the metadata for a chain and then removes all sub-chains from the database
def delete_sub_chains(chain_metadata):
    for meta in chain_metadata[1:]:
        delete_email(meta['from'], meta['to'], meta['date'], meta['subject'])


#Example
email_chain = """From: Jack
Sent: Wednesday, January 12, 2021 10:09 AM
To: Jill
Subject: Tree House
This is the body of the email.

From: Jill
Sent: Tuesday, January 11, 2021 11:15 AM
To: Jack
Subject: Tree House
I want to build a tree house."""

#Split chain into list of email bodies, list of corresponding metadata
emails, metadata = split_email_chain(email_chain)

#Print results to make sure splitting is done correctly
print("Emails: ")
for email, meta in zip(emails, metadata):
    print(f"From: {meta['from']}, To: {meta['to']}, Sent: {meta['date']}, Subject: {meta['subject']}")
    print("Email Body:", email)
    print("-" * 80)

#Print chain as conversation/script
print('\n')
print('Conversation: ')
conversation = format_chain_as_conversation(metadata,emails)
print(conversation)