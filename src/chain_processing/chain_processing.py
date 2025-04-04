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
    latest_email_metadata = metadata_list[0]
    latest_email_header = f"{latest_email_metadata['subject']} ({latest_email_metadata['date']})\n"

    #Reverse order of emails in chain, so that goes from earliest email to latest email now
    reversed_metadata_list = metadata_list[::-1]
    reversed_bodies = email_bodies[::-1]

    #List to use to format chain as conversation
    formatted_parts = []
    i = 0

    while i < len(reversed_metadata_list):
        #Get metadata and body of current email
        metadata = reversed_metadata_list[i]
        body = reversed_bodies[i]

        #Check if this email is forwarding another, if so, deal with this special case
        if ("fwd:" in metadata["subject"].lower() or "fw:" in metadata["subject"].lower()) and ((i - 1) >= 0):
            prev_email_metadata = reversed_metadata_list[i - 1]
            #Need to make sure the next email in chain is truly a forwarded email and not a reply (make sure sender of that one differs from receiver of this one)
            if prev_email_metadata['from'] != metadata['to']:
               #Since the next email is being forwarded by this one, we may have "Original Message" in our body prior to this email; we should remove this
               body = re.sub(r"\nOriginal Message\n", "", body, flags=re.IGNORECASE).strip()
               quoted_body = reversed_bodies[i - 1]  #Body of the forwarded message

            #If Jack says to Jill, "Look at this" and forwards an email from John which said "I think cats are better than dogs",
            #we'll add to our script:
            #Jack: {Jill, Look at this. John said: {I think cats are better than dogs.}}
            quoted_body = f'{{{quoted_body}}}' #add brackets around quoted body
            body = f"{metadata['to']}, {body} {prev_email_metadata['from']} said: {quoted_body}" #Format as a line in the script
            reversed_bodies[i]=body #Edit the stored body in case this email is forwarded to the next in the chain

        #If we are not forwarding, it's simpler. We put the body in brackets and add to the script,
        #Sender: {body}
        body = f'{{{body}}}'
        formatted_parts.append(f"{metadata['from']}: {body}")
        i += 1

    #Combine the header (subject, date) and the conversation script
    return latest_email_header + "\n".join(formatted_parts)

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

#Examples
print("Example 1")
print("_________")
email_chain = """From: Jack
Sent: Wednesday, January 12, 2021 10:09 AM
To: Jill
Subject: Tree House
That sounds like a great idea.

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
print('Conversation: ')
conversation = format_chain_as_conversation(metadata,emails)
print(conversation)

print("\nExample 2")
print("_________")
email_chain = """From: Jack
Sent: Wednesday, January 12, 2021 10:09 AM
To: Jill
Subject: Fwd: Get a dog
We should get a dog.

From: John
Sent: Tuesday, January 11, 2021 11:15 AM
To: Jack
Subject: Get a dog
You should get a dog. Mine is great."""

#Split chain into list of email bodies, list of corresponding metadata
emails, metadata = split_email_chain(email_chain)
#Print results to make sure splitting is done correctly
print("Emails: ")
for email, meta in zip(emails, metadata):
    print(f"From: {meta['from']}, To: {meta['to']}, Sent: {meta['date']}, Subject: {meta['subject']}")
    print("Email Body:", email)
    print("-" * 80)

#Print chain as conversation/script
print('Conversation: ')
conversation = format_chain_as_conversation(metadata,emails)
print(conversation)

print("\nExample 3")
print("_________")
email_chain = """From: Jill
Sent: Wednesday, January 12, 2021 10:09 AM
To: Jane
Subject: Fwd: Fwd: Get a dog
Look at this.

From: Jack
Sent: Wednesday, January 12, 2021 10:09 AM
To: Jill
Subject: Fwd: Get a dog
We should get a dog.

From: John
Sent: Tuesday, January 11, 2021 11:15 AM
To: Jack
Subject: Get a dog
You should get a dog. Mine is great."""

#Split chain into list of email bodies, list of corresponding metadata
emails, metadata = split_email_chain(email_chain)
#Print results to make sure splitting is done correctly
print("Emails: ")
for email, meta in zip(emails, metadata):
    print(f"From: {meta['from']}, To: {meta['to']}, Sent: {meta['date']}, Subject: {meta['subject']}")
    print("Email Body:", email)
    print("-" * 80)

#Print chain as conversation/script
print('Conversation: ')
conversation = format_chain_as_conversation(metadata,emails)
print(conversation)