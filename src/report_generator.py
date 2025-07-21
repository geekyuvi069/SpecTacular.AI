import os
import json
import csv

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_test_cases(test_cases, file_prefix="generated_test_cases"):
    """
    Saves test cases in both JSON and CSV formats.

    Args:
        test_cases: List of dictionaries with keys like title, description, steps, expected
    """
    json_path = os.path.join(OUTPUT_DIR, f"{file_prefix}.json")
    csv_path = os.path.join(OUTPUT_DIR, f"{file_prefix}.csv")

    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(test_cases, f, indent=4, ensure_ascii=False)

    # Save CSV
    if test_cases:
        keys = test_cases[0].keys()
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(test_cases)

    print(f"✅ Test cases saved to: {json_path} and {csv_path}")

def generate_traceability_matrix(requirements, test_cases, file_prefix="traceability_matrix"):
    """
    Creates a CSV file mapping requirements to test case titles.

    Args:
        requirements: List of requirement strings (e.g., unmatched SRS chunks)
        test_cases: List of dicts with at least 'title'
    """
    matrix_path = os.path.join(OUTPUT_DIR, f"{file_prefix}.csv")
    with open(matrix_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Requirement", "Test Case Title"])
        for req, tc in zip(requirements, test_cases):
            writer.writerow([req, tc.get("title", "Unnamed")])

    print(f"✅ Traceability Matrix saved to: {matrix_path}")
