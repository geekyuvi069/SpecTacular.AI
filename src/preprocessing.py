import re
import json
import csv
from PyPDF2 import PdfReader

def extract_text_from_pdf(file_path):
    """
    Extracts raw text from a PDF SRS document.
    """
    text = ""
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def clean_text(text):
    """
    Cleans extracted SRS text by normalizing spaces and removing extra newlines.
    """
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def split_into_chunks(text, chunk_size=200):
    """
    Splits cleaned SRS text into chunks of N words (for NLP/model input).
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def load_test_cases(file_path):
    """
    Loads previously generated test cases from a JSON or CSV file.
    Returns a list of dictionaries with keys like 'requirement' and 'test_case'.
    """
    test_cases = []

    if file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            test_cases = json.load(f)

    elif file_path.endswith(".csv"):
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                test_cases.append(row)

    else:
        raise ValueError("Unsupported file type for test cases. Use .json or .csv")

    return test_cases

def process_inputs(srs_path, test_case_path):
    """
    Full pipeline to:
    - Extract and clean SRS
    - Split into chunks
    - Load existing test cases

    Returns: (chunks, test_case_list)
    """
    raw_text = extract_text_from_pdf(srs_path)
    cleaned_text = clean_text(raw_text)
    srs_chunks = split_into_chunks(cleaned_text)

    test_cases = load_test_cases(test_case_path)

    return srs_chunks, test_cases

