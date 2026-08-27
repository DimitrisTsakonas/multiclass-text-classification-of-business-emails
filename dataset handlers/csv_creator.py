"""
Purpose:
    Parses email files (.eml) and text files (.txt) within a directory and
    its subdirectories, extracting subject, sender, receiver, CC, date and
    body from each, then writes the results to a CSV file.

Thesis reference:
    Section 4.2.1 - Dataset. Produces the per-month CSV files (Version 4)
    that later stages clean and convert into the final dataset structure.

Inputs:
    A directory of .eml/.txt files (emails_directory).

Outputs:
    A CSV file (output_csv) with columns: Label, EmailSoftware_ID, Date,
    Incoming, Outgoing, From, To, CC, Subject, Body.

Notes:
    This script was run manually once per month (12 times total) to build
    the full year of data - emails_directory and output_csv were edited by
    hand each time to point at that month's folder before running.
"""

import os
import csv
from email.parser import BytesParser
import base64
import time


def parse_email(file_path):
    """This function takes the file path of an email or text file as input.
    For .txt files:
        Subject: between "Daily Update for EmailSoftware" and "- EmailSoftware".
        Date: From the subject line.
        Sender: Set to "www.emailsoftware.example", Receiver: Set to "chartering@shipbrokingcompany.example", and CC to 0.
        Body: between "The following is a summary recap" and "The leader in Maritime Message Management".
    For .eml files:
        It parses the email using the BytesParser from the email.parser module.
        Extracts the subject, sender, receiver, CC, date, and body from the email content.
        Returns the extracted information as a tuple: (subject, sender, receiver, CC, date, body)."""

    with open(file_path, 'rb') as file:
        if file_path.endswith('.txt'):
            # Read the file content
            content = file.read().decode('utf-8', errors='ignore')
            # Find the start and end positions of the subject line
            start_index = content.find('Daily Update for EmailSoftware')
            end_index = content.find('- EmailSoftware') + len('- EmailSoftware')
            # Extract the subject line
            subject = content[start_index:end_index].strip()
            # Extract the date from the subject line
            date_start_index = subject.find('Shipbroking Suite') + len('Shipbroking Suite') + 1
            date_end_index = subject.find(' - EmailSoftware')
            date = subject[date_start_index:date_end_index].strip()
            # Set other fields to empty strings
            sender = 'www.emailsoftware.example'
            receiver = 'chartering@shipbrokingcompany.example'
            cc = ""

            # Extract body between specified strings
            body_start_index = content.find('The following is a summary recap')
            body_end_index = content.find('The leader in Maritime Message Management') + len(
                'The leader in Maritime Message Management')
            body = content[body_start_index:body_end_index].strip()
        else:
            # Parse the email using BytesParser for .eml files
            msg = BytesParser().parse(file)
            subject = msg.get('Subject', '')
            sender = msg.get('From', '')
            receiver = msg.get('To', '')
            cc = msg.get('Cc', '')
            date = msg.get('Date', '')
            body = ''
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    if content_type == 'text/plain':
                        encoding = part.get('Content-Transfer-Encoding', '')
                        if encoding.lower() == 'base64':
                            try:
                                body = base64.b64decode(part.get_payload()).decode('utf-8')
                            except UnicodeDecodeError:
                                body = base64.b64decode(part.get_payload()).decode('latin-1', errors='ignore')
                        else:
                            try:
                                body = part.get_payload(decode=True).decode('utf-8')
                            except UnicodeDecodeError:
                                body = part.get_payload(decode=True).decode('latin-1', errors='ignore')
                        break
            else:
                encoding = msg.get('Content-Transfer-Encoding', '')
                if encoding.lower() == 'base64':
                    try:
                        body = base64.b64decode(msg.get_payload()).decode('utf-8')
                    except UnicodeDecodeError:
                        body = base64.b64decode(msg.get_payload()).decode('latin-1', errors='ignore')
                else:
                    try:
                        body = msg.get_payload(decode=True).decode('utf-8')
                    except UnicodeDecodeError:
                        body = msg.get_payload(decode=True).decode('latin-1', errors='ignore')
    return subject, sender, receiver, cc, date, body


def save_to_csv(emails_directory, output_csv):
    """Input: directory containing email/text files, Output: CSV file path.

    Iterates through all files in the specified directory and its subdirectories.
    Incoming/Outgoing field is set according to the directory name of the file.
    Sets the Label field based on the last character of the parent folder name.
    Calls parse_email() to extract information from each file.
    Writes the extracted information to the CSV file.
    Note that escapechar='\\' is set.
    (Label, CC, Incoming/Outgoing flags, EmailSoftware_ID, subject, sender, receiver, date, body) """
    start_time = time.time()
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["Label", 'EmailSoftware_ID', 'Date', "Incoming", "Outgoing", 'From', 'To', 'CC', 'Subject', 'Body']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, escapechar='\\')
        writer.writeheader()
        rows = 0
        for root, _, files in os.walk(emails_directory):
            for filename in files:
                if filename.endswith('.eml') or filename.endswith('.txt'):
                    file_path = os.path.join(root, filename)
                    emailsoftware_id = os.path.splitext(filename)[0]  # Extracting file name without extension
                    emailsoftware_id = emailsoftware_id.replace('_', '')  # Replace underscores with dashes if needed

                    # Determine if the file is in an incoming or outgoing directory
                    incoming = 1 if "incoming" in root.lower() else 0
                    outgoing = 1 if "outgoing" in root.lower() else 0
                    # Set label according to very last folder name, last digit
                    label = os.path.basename(os.path.dirname(file_path))[-1]

                    subject, sender, receiver, cc, date, body = parse_email(file_path)
                    writer.writerow(
                        {'Label': label, 'CC': cc, 'Incoming': incoming, 'Outgoing': outgoing, 'EmailSoftware_ID': emailsoftware_id,
                         'Subject': subject, 'From': sender, 'To': receiver, 'Date': date, 'Body': body})
                    rows += 1

                    print(f"Current csv rows: {rows}")

        print(f"Total csv rows: {rows}")
        end_time = time.time()
        print(f"Saving to CSV took {end_time - start_time} seconds.")
        print(f"Saving to CSV took {(end_time - start_time)/60} minutes.")


emails_directory = "<path-to-dataset>/Version 2/December_23"
output_csv = "<path-to-dataset>/Version 4 monthly csv/12_December_23_version_4.csv"
save_to_csv(emails_directory, output_csv)