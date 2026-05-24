import json
from pathlib import Path
from typing import Any

from openpyxl import Workbook, load_workbook
from tqdm import tqdm
from dotenv import load_dotenv
from query_processor import QueryProcessor
from llm_factory import create_llm


def load_json_list(file_path: str | Path) -> list[dict[str, Any]]:
    path = Path(file_path)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON value to be a list.")

    # Optional: ensure each item is a JSON object (Python dict)
    if not all(isinstance(item, dict) for item in data):
        raise ValueError("Expected every item in the list to be a JSON object.")

    return data


def get_or_create_workbook(excel_path: Path):
    if excel_path.exists():
        workbook = load_workbook(excel_path)
        sheet = workbook.active
    else:
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "results"
        sheet.append(["input", "context", "actual_output", "expected_output", "routed_agent"])
        workbook.save(excel_path)
    return workbook, sheet


items = load_json_list("qafinalclean.json")
print(f"Loaded {len(items)} records")
if not items:
    raise ValueError("No records found in qafinalclean.json")


load_dotenv()

queryProcessor = QueryProcessor(
    create_llm(),
    verbose = False
    )

excel_file = Path("deep_test_results.xlsx")
workbook, sheet = get_or_create_workbook(excel_file)

# Resume from the first unprocessed item based on rows already persisted.
processed_count = max(sheet.max_row - 1, 0)
if processed_count >= len(items):
    print("All items are already processed. Nothing to do.")
else:
    print(f"Resuming from item index {processed_count}")

for index in tqdm(range(processed_count, len(items)), desc="Processing queries", unit="query"):
    item = items[index]
    query_input = item.get("input", "")
    expected_output = item.get("output", item.get("expected_output", ""))

    try:
        actual_output = queryProcessor.processQuery(query_input, [])
        context = queryProcessor.get_generation_context()
        routed_agent = queryProcessor.get_route_decision()
    except Exception as exc:
        actual_output = f"ERROR: {exc}"
        context = ""
        routed_agent = ""

    sheet.append([
        query_input,
        context,
        actual_output,
        expected_output,
        routed_agent,
    ])
    workbook.save(excel_file)

workbook.close()
print(f"Completed. Results are in {excel_file}")
