"""
=============================================================================
Expense Tracker App - Dataset Validation & Health Check Script
=============================================================================
This script loads and validates the expense dataset (`data/expenses.csv`).
It performs data quality, schema verification, and integrity checks:
1. Verifies column presence against the required schema.
2. Checks for missing / null values across all columns.
3. Identifies and counts duplicate rows.
4. Checks column data types and tests date format parsing.
5. Verifies numeric integrity (e.g. non-positive or negative amounts).
6. Generates a clear validation summary report in `outputs/dataset_validation.txt`.
=============================================================================
"""

import os
import sys
import pandas as pd


def validate_expense_dataset(file_path: str, output_report_path: str) -> dict:
    """
    Performs comprehensive data quality validation on the expense CSV file.

    Parameters:
        file_path (str): Relative or absolute path to the dataset CSV.
        output_report_path (str): Destination path for the text validation report.

    Returns:
        dict: Summary dictionary containing validation results.
    """
    print("=" * 70)
    print(" EXPENSE TRACKER DATASET VALIDATION")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Check File Existence & Load Dataset
    # -------------------------------------------------------------------------
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The dataset was not found at '{file_path}'.")

    print(f"[*] Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)

    # -------------------------------------------------------------------------
    # 2. Basic Dimensions (Rows & Columns)
    # -------------------------------------------------------------------------
    num_rows, num_cols = df.shape
    print(f"[+] Loaded successfully: {num_rows} rows, {num_cols} columns")

    # -------------------------------------------------------------------------
    # 3. Schema & Required Columns Verification
    # -------------------------------------------------------------------------
    required_columns = ["Date", "Category", "Description", "Amount", "Payment_Method", "Type"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    extra_columns = [col for col in df.columns if col not in required_columns]

    schema_valid = len(missing_columns) == 0

    # -------------------------------------------------------------------------
    # 4. Missing / Null Values Check
    # -------------------------------------------------------------------------
    null_counts = df.isnull().sum()
    total_nulls = int(null_counts.sum())

    # -------------------------------------------------------------------------
    # 5. Duplicate Rows Check
    # -------------------------------------------------------------------------
    duplicate_count = int(df.duplicated().sum())

    # -------------------------------------------------------------------------
    # 6. Data Types & Date Parsing Check
    # -------------------------------------------------------------------------
    dtypes_dict = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Test date parsing
    date_parse_errors = 0
    try:
        parsed_dates = pd.to_datetime(df["Date"], errors="coerce")
        date_parse_errors = int(parsed_dates.isnull().sum())
        min_date = str(parsed_dates.min().date())
        max_date = str(parsed_dates.max().date())
    except Exception as e:
        date_parse_errors = -1
        min_date = "N/A"
        max_date = "N/A"

    # -------------------------------------------------------------------------
    # 7. Amount Field Validity (Invalid or Non-Positive Amounts)
    # -------------------------------------------------------------------------
    amount_numeric_errors = 0
    non_positive_amounts = 0
    try:
        numeric_amounts = pd.to_numeric(df["Amount"], errors="coerce")
        amount_numeric_errors = int(numeric_amounts.isnull().sum())
        non_positive_amounts = int((numeric_amounts <= 0).sum())
        min_amount = float(numeric_amounts.min())
        max_amount = float(numeric_amounts.max())
    except Exception:
        amount_numeric_errors = -1
        min_amount = 0.0
        max_amount = 0.0

    # -------------------------------------------------------------------------
    # 8. Distinct Categories & Types Breakdown
    # -------------------------------------------------------------------------
    category_counts = df["Category"].value_counts().to_dict()
    type_counts = df["Type"].value_counts().to_dict()
    payment_method_counts = df["Payment_Method"].value_counts().to_dict()

    # -------------------------------------------------------------------------
    # 9. Format First 10 Rows
    # -------------------------------------------------------------------------
    first_10_rows = df.head(10)
    print("\n" + "-" * 70)
    print(" FIRST 10 ROWS OF THE DATASET:")
    print("-" * 70)
    print(first_10_rows.to_string(index=False))

    # -------------------------------------------------------------------------
    # 10. Generate Report Text
    # -------------------------------------------------------------------------
    report_lines = [
        "=" * 70,
        "EXPENSE TRACKER APP - DATASET VALIDATION REPORT",
        "=" * 70,
        f"File Path Analyzed     : {file_path}",
        f"Total Rows             : {num_rows}",
        f"Total Columns          : {num_cols}",
        "",
        "1. SCHEMA VERIFICATION:",
        f"   - Required Columns  : {required_columns}",
        f"   - Present Columns   : {list(df.columns)}",
        f"   - Schema Status     : {'PASSED (All required columns found)' if schema_valid else 'FAILED (Missing: ' + str(missing_columns) + ')'}",
        "",
        "2. MISSING VALUES ANALYSIS:",
        f"   - Total Missing/NaN : {total_nulls}",
    ]

    for col in df.columns:
        report_lines.append(f"     * {col:<16} : {null_counts[col]} missing values")

    report_lines.extend([
        "",
        "3. DUPLICATE ROWS ANALYSIS:",
        f"   - Duplicate Rows    : {duplicate_count}",
        f"   - Status            : {'PASSED (Zero duplicates detected)' if duplicate_count == 0 else 'WARNING (Duplicates found)'}",
        "",
        "4. DATA TYPES & PARSING:",
    ])

    for col, dtype in dtypes_dict.items():
        report_lines.append(f"     * {col:<16} : {dtype}")

    report_lines.extend([
        f"   - Date Parsing      : {'PASSED (Valid YYYY-MM-DD)' if date_parse_errors == 0 else 'ERRORS: ' + str(date_parse_errors)}",
        f"   - Date Span         : {min_date} to {max_date}",
        "",
        "5. AMOUNT FIELD INTEGRITY:",
        f"   - Non-numeric / NaN : {amount_numeric_errors}",
        f"   - Non-positive (<=0): {non_positive_amounts}",
        f"   - Min Amount        : Rs. {min_amount:,.2f}",
        f"   - Max Amount        : Rs. {max_amount:,.2f}",
        f"   - Status            : {'PASSED (All amounts are strictly positive)' if (amount_numeric_errors == 0 and non_positive_amounts == 0) else 'FAILED'}",
        "",
        "6. CATEGORY & TYPE DISTRIBUTIONS:",
        f"   - Transaction Types : {type_counts}",
        f"   - Payment Methods   : {payment_method_counts}",
        f"   - Categories ({len(category_counts)}):",
    ])

    for cat, count in category_counts.items():
        report_lines.append(f"     * {cat:<16} : {count} transactions")

    overall_status = "HEALTHY & READY FOR ANALYSIS" if (
        schema_valid and total_nulls == 0 and duplicate_count == 0 and non_positive_amounts == 0
    ) else "NEEDS CLEANING"

    report_lines.extend([
        "",
        "=" * 70,
        f"OVERALL DATASET HEALTH: {overall_status}",
        "=" * 70,
    ])

    report_content = "\n".join(report_lines)

    # -------------------------------------------------------------------------
    # 11. Write Report to outputs/dataset_validation.txt
    # -------------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print("\n" + "-" * 70)
    print(f"[+] Validation report successfully written to: {output_report_path}")
    print(f"[*] Overall Health Status: {overall_status}")
    print("=" * 70)

    return {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "schema_valid": schema_valid,
        "total_nulls": total_nulls,
        "duplicate_count": duplicate_count,
        "non_positive_amounts": non_positive_amounts,
        "min_date": min_date,
        "max_date": max_date,
        "overall_status": overall_status,
        "report_content": report_content
    }


def main():
    """
    Main entry point for command-line execution.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))

    dataset_path = os.path.join(project_root, "data", "expenses.csv")
    report_path = os.path.join(project_root, "outputs", "dataset_validation.txt")

    try:
        validate_expense_dataset(dataset_path, report_path)
    except Exception as err:
        print(f"[!] Validation failed with error: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
