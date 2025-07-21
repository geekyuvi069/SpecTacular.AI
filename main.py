import os
from datetime import datetime
from src.preprocessing import process_inputs
from src.vocabulary import Vocabulary
from src.encoder import TransformerEncoder
from src.decoder import TransformerDecoder
from src.semantic_search import match_srs_to_testcases

DATA_FOLDER = "data"

def check_uploaded_files():
    """
    Checks for /data/srs.pdf and /data/old_tests.json for upload_dual endpoint
    """
    srs_path = os.path.join(DATA_FOLDER, "srs.pdf")
    test_path = os.path.join(DATA_FOLDER, "old_tests.json")
    return os.path.exists(srs_path) and os.path.exists(test_path)

def get_latest_srs_and_testcase_files():
    """
    For /upload route with date-based folders
    Returns: Tuple (srs_file_path, testcases_file_path)
    """
    if not os.path.exists(DATA_FOLDER):
        raise FileNotFoundError(f"'{DATA_FOLDER}' directory does not exist.")

    # List subfolders by most recent
    subfolders = sorted(
        [os.path.join(DATA_FOLDER, f) for f in os.listdir(DATA_FOLDER)
         if os.path.isdir(os.path.join(DATA_FOLDER, f))],
        reverse=True
    )

    for folder in subfolders:
        srs_file = None
        testcases_file = None

        for fname in os.listdir(folder):
            if fname.lower().endswith(".pdf"):
                srs_file = os.path.join(folder, fname)
            elif fname.lower().endswith(".json") or fname.lower().endswith(".csv"):
                testcases_file = os.path.join(folder, fname)

        if srs_file and testcases_file:
            return srs_file, testcases_file

    raise FileNotFoundError("Could not find both an SRS PDF and test case file in recent uploads.")

def run_console_processing(srs_path, tc_path):
    print(f"📄 SRS File: {srs_path}")
    print(f"📋 Test Case File: {tc_path}")

    srs_chunks, existing_test_cases = process_inputs(srs_path, tc_path)

    print(f"✅ Extracted {len(srs_chunks)} SRS chunks")
    print(f"✅ Loaded {len(existing_test_cases)} test cases")

    unmatched_chunks = match_srs_to_testcases(srs_chunks, existing_test_cases)
    print(f"❗ {len(unmatched_chunks)} unmatched requirements found")

    # Optional: Build vocab & initialize model
    if unmatched_chunks:
        vocab = Vocabulary()
        vocab.build_vocab(unmatched_chunks)
        print(f"🔠 Vocabulary has {len(vocab.word_to_id)} tokens")

        encoder = TransformerEncoder(len(vocab.word_to_id), 256, 8, 6)
        decoder = TransformerDecoder(len(vocab.word_to_id), 256, 8, 6)
        print("🧠 Model initialized for unmatched requirements")

    print("\n🟢 Done.")

if __name__ == "__main__":
    print("=== SmartSpec AI - Console Processor ===")

    if check_uploaded_files():
        print("✅ Using upload_dual files")
        srs_path = os.path.join(DATA_FOLDER, "srs.pdf")
        tc_path = os.path.join(DATA_FOLDER, "old_tests.json")
        run_console_processing(srs_path, tc_path)
    else:
        try:
            print("⚠️ Dual upload not found, checking recent uploads...")
            srs_path, tc_path = get_latest_srs_and_testcase_files()
            run_console_processing(srs_path, tc_path)
        except FileNotFoundError as e:
            print(f"❌ {e}")
