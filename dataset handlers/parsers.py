"""
Purpose:
    Parses a raw .eml file and tags it with custom metadata headers used
    throughout the pipeline (LABEL, INCOMING, OUTGOING, EMAILSOFTWARE_ID),
    then re-saves the tagged message as a new .eml file.

Thesis reference:
    Section 4.2.1 - Dataset. Part of the raw email preparation stage that
    produces the labeled dataset structure (Sender, Receiver, CC, Subject,
    Body) described in the dataset construction process.

Inputs:
    A raw .eml file path, passed to eml_parser().

Outputs:
    eml_parser() returns the tagged email.message object and its filename.
    eml_saver() writes that message to disk as <file_name>.eml under a
    'created_emails/' folder.
"""

from email import policy
from email.parser import BytesParser
import email
import os


def eml_saver(msg, file_name):
    """Takes output from parser and then creates keys + values and saves it as .eml"""
    # Reconstruct the email message
    new_msg = email.message.EmailMessage()

    iterator = 0
    for key in msg.keys():
        new_msg[key] = msg.values()[iterator]  # creating keys and their values
        iterator += 1

    new_msg.set_payload(msg.get_payload())  # adding the email body and attachments to the new message

    # Save the new email as a .eml file
    with open(f'created_emails/{file_name}.eml', 'wb') as new_fp:
        new_fp.write(new_msg.as_bytes())

    print("created.")


def eml_parser(eml):
    """This function parses an .eml file and modifies its headers.
    Custom headers: ("LABEL", "INCOMING", "OUTGOING", "EMAILSOFTWARE_ID")"""
    with open(eml, 'rb') as fp:
        path_name = os.path.basename(eml)
        file_name = os.path.splitext(path_name)[0]

        msg = BytesParser(policy=policy.default).parse(fp)
        msg.add_header("LABEL", "U")
        msg.add_header("INCOMING", "1")
        msg.add_header("OUTGOING", "0")
        msg.add_header("EMAILSOFTWARE_ID", file_name)

    return msg, file_name


