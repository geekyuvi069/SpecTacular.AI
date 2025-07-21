from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from datetime import datetime

from src.preprocessing import process_inputs
from src.semantic_search import build_index, search_unmatched_requirements

app = Flask(__name__, static_folder="web", static_url_path="")
CORS(app)

srs_chunks = []
existing_test_cases = []

UPLOAD_FOLDER = "./data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return send_from_directory("web", "index.html")

@app.route("/upload", methods=["POST"])
def upload():
    global srs_chunks, existing_test_cases

    if "srs_file" not in request.files or "testcases_file" not in request.files:
        return jsonify({"error": "Both SRS and test case files are required."}), 400

    srs_file = request.files["srs_file"]
    testcases_file = request.files["testcases_file"]

    if srs_file.filename == "" or testcases_file.filename == "":
        return jsonify({"error": "Missing file name(s)."}), 400

    # Save files to date-based folder
    date_folder = datetime.now().strftime("%Y-%m-%d")
    save_folder = os.path.join(UPLOAD_FOLDER, date_folder)
    os.makedirs(save_folder, exist_ok=True)

    srs_path = os.path.join(save_folder, srs_file.filename)
    tc_path = os.path.join(save_folder, testcases_file.filename)

    srs_file.save(srs_path)
    testcases_file.save(tc_path)

    srs_chunks, existing_test_cases = process_inputs(srs_path, tc_path)
    build_index(srs_chunks, existing_test_cases)

    return jsonify({
        "message": "Both files processed successfully.",
        "total_chunks": len(srs_chunks),
        "total_existing_test_cases": len(existing_test_cases)
    })

@app.route("/upload_dual", methods=["POST"])
def upload_dual():
    if "srs" not in request.files or "old_tests" not in request.files:
        return jsonify({'error': 'Both SRS and old test cases are required.'}), 400

    srs_file = request.files["srs"]
    old_tests_file = request.files["old_tests"]

    if srs_file.filename == "" or old_tests_file.filename == "":
        return jsonify({'error': 'Empty filenames not allowed.'}), 400

    srs_path = os.path.join(UPLOAD_FOLDER, "srs.pdf")
    test_path = os.path.join(UPLOAD_FOLDER, "old_tests.json")

    srs_file.save(srs_path)
    old_tests_file.save(test_path)

    return jsonify({'message': 'Files uploaded successfully.'})

@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    query_text = data.get("query", "").strip()

    if not query_text:
        return jsonify({"error": "Empty query."}), 400

    unmatched_chunks = search_unmatched_requirements(query_text)

    test_cases = []
    for i, chunk in enumerate(unmatched_chunks, 1):
        test_cases.append({
            "title": f"Generated Test Case {i}",
            "description": "Generated from unmatched requirement.",
            "steps": chunk,
            "expected": "Functionality as defined by requirement."
        })

    return jsonify({
        "query": query_text,
        "testCases": test_cases
    })

if __name__ == "__main__":
    app.run(debug=True)
