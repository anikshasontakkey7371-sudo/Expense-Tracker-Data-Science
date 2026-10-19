"""
=============================================================================
Expense Tracker App - Data Cleaning & Preprocessing Pipeline
=============================================================================
This module is Phase 2 of the Data Science pipeline.
It takes raw financial data (`data/expenses.csv`) and produces a standardized,
cleaned, and validated dataset saved at `data/cleaned_expenses.csv`.

Pipeline Steps:
1. File Existence & Loading: Loads raw CSV into Pandas DataFrame.
2. Data Type Conversion:
   - Converts 'Date' into datetime objects.
   - Converts 'Amount' into float numeric values.
3. Text Standardization:
   - Strips leading/trailing whitespaces.
   - Normalizes text casing (e.g. 'UPI', 'Credit Card', 'Food', 'Income').
4. Missing & Duplicate Handling:
   - Identifies and counts missing values.
   - Removes any duplicate records while preserving the first instance.
5. Business Rule Validation:
   - Validates that 'Amount' is strictly positive (> 0).
   - Validates that 'Type' is strictly 'Income' or 'Expense'.
   - Removes unparseable dates or corrupt rows.
6. Export & Report:
   - Exports cleaned data to `data/cleaned_expenses.csv`.
   - Generates a detailed audit report in `outputs/cleaning_report.txt`.
=============================================================================
"""

import os
import sys
import pandas as pd


def standardize_payment_method(val: str) -> str:
    """
    Standardizes payment method strings to consistent casing:
    'upi' / 'UPI ' -> 'UPI'
    'credit card'  -> 'Credit Card'
    'debit card'   -> 'Debit Card'
    'net banking'  -> 'Net Banking'
    'cash'         -> 'Cash'
    """
    if pd.isna(val):
        return val
    cleaned = str(val).strip()
    if cleaned.upper() == "UPI":
        return "UPI"
    return cleaned.title()


def clean_expense_data(
    input_csv_path: str,
    output_csv_path: str,
    output_report_path: str
) -> pd.DataFrame:
    """
    Executes the full cleaning and validation pipeline on the expenses dataset.

    Parameters:
        input_csv_path (str): Path to raw CSV file.
        output_csv_path (str): Destination path for cleaned CSV.
        output_report_path (str): Destination path for the text audit report.

    Returns:
        pd.DataFrame: Cleaned and validated DataFrame.
    """
    print("=" * 70)
    print(" EXPENSE TRACKER - DATA CLEANING PIPELINE")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Confirm File Existence & Load
    # -------------------------------------------------------------------------
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(
            f"Error: Input dataset not found at '{input_csv_path}'. "
            "Please run 'python data/generate_data.py' first."
        )

    print(f"[*] Step 1: Loading raw dataset from '{input_csv_path}'...")
    df_raw = pd.read_csv(input_csv_path)
    initial_rows = len(df_raw)
    initial_cols = len(df_raw.columns)
    print(f"[+] Loaded {initial_rows} rows and {initial_cols} columns.")

    # -------------------------------------------------------------------------
    # 2. Audit Missing Values in Raw Data
    # -------------------------------------------------------------------------
    initial_nulls = df_raw.isnull().sum().to_dict()
    total_initial_nulls = int(df_raw.isnull().sum().sum())
    print(f"[*] Step 2: Checking missing values: {total_initial_nulls} total missing cells found.")

    # -------------------------------------------------------------------------
    # 3. Audit Duplicate Rows in Raw Data
    # -------------------------------------------------------------------------
    initial_duplicates = int(df_raw.duplicated().sum())
    print(f"[*] Step 3: Checking duplicate rows: {initial_duplicates} exact duplicate rows found.")

    # Create working copy for transformation
    df_clean = df_raw.copy()

    # -------------------------------------------------------------------------
    # 4. Standardize Text Columns (Strip whitespace & casing)
    # -------------------------------------------------------------------------
    print("[*] Step 4: Standardizing text formatting across string columns...")

    # Standardize Category -> Title case (e.g. 'Food', 'Healthcare')
    if "Category" in df_clean.columns:
        df_clean["Category"] = df_clean["Category"].astype(str).str.strip().str.title()

    # Standardize Description -> Trim whitespace
    if "Description" in df_clean.columns:
        df_clean["Description"] = df_clean["Description"].astype(str).str.strip()

    # Standardize Type -> Capitalized ('Income' or 'Expense')
    if "Type" in df_clean.columns:
        df_clean["Type"] = df_clean["Type"].astype(str).str.strip().str.capitalize()

    # Standardize Payment_Method -> Title case, preserving acronym 'UPI'
    if "Payment_Method" in df_clean.columns:
        df_clean["Payment_Method"] = df_clean["Payment_Method"].apply(standardize_payment_method)

    # -------------------------------------------------------------------------
    # 5. Convert Data Types
    # -------------------------------------------------------------------------
    print("[*] Step 5: Converting data types (Date -> Datetime, Amount -> Float)...")

    # Date conversion with errors='coerce' to flag unparseable dates
    df_clean["Date_Parsed"] = pd.to_datetime(df_clean["Date"], errors="coerce")
    invalid_dates_count = int(df_clean["Date_Parsed"].isnull().sum())

    # Amount conversion with errors='coerce' to flag non-numeric values
    df_clean["Amount_Parsed"] = pd.to_numeric(df_clean["Amount"], errors="coerce")
    invalid_amount_types_count = int(df_clean["Amount_Parsed"].isnull().sum())

    # -------------------------------------------------------------------------
    # 6. Filter Invalid Records (Business Logic Rules)
    # -------------------------------------------------------------------------
    print("[*] Step 6: Validating business rules and filtering invalid records...")

    # Rule 1: Date must be valid
    valid_date_mask = df_clean["Date_Parsed"].notnull()

    # Rule 2: Amount must be numeric and strictly positive (> 0)
    valid_amount_mask = (df_clean["Amount_Parsed"].notnull()) & (df_clean["Amount_Parsed"] > 0)

    # Rule 3: Type must be either 'Income' or 'Expense'
    valid_type_mask = df_clean["Type"].isin(["Income", "Expense"])

    # Count violations for the report
    non_positive_amount_count = int((df_clean["Amount_Parsed"] <= 0).sum())
    invalid_type_count = int((~valid_type_mask).sum())

    # Combined valid mask
    all_valid_mask = valid_date_mask & valid_amount_mask & valid_type_mask
    invalid_records_count = int((~all_valid_mask).sum())

    # Filter invalid records
    df_clean = df_clean[all_valid_mask].copy()

    # Update columns with validated parsed versions
    df_clean["Date"] = df_clean["Date_Parsed"].dt.strftime("%Y-%m-%d")
    df_clean["Amount"] = df_clean["Amount_Parsed"].round(2)
    df_clean.drop(columns=["Date_Parsed", "Amount_Parsed"], inplace=True)

    # -------------------------------------------------------------------------
    # 7. Deduplicate Records
    # -------------------------------------------------------------------------
    print("[*] Step 7: Deduplicating dataset...")
    rows_before_dedup = len(df_clean)
    df_clean.drop_duplicates(keep="first", inplace=True)
    duplicates_removed = rows_before_dedup - len(df_clean)

    # -------------------------------------------------------------------------
    # 8. Sort Chronologically
    # -------------------------------------------------------------------------
    df_clean.sort_values(by=["Date", "Type"], ascending=[True, False], inplace=True)
    df_clean.reset_index(drop=True, inplace=True)
    final_rows = len(df_clean)

    # -------------------------------------------------------------------------
    # 9. Save Cleaned Dataset to CSV
    # -------------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_clean.to_csv(output_csv_path, index=False)
    print(f"[+] Step 8: Cleaned dataset saved to: '{output_csv_path}'")

    # -------------------------------------------------------------------------
    # 10. Generate Audit Report
    # -------------------------------------------------------------------------
    print("[*] Step 9: Compiling data cleaning audit report...")
    report_lines = [
        "=" * 70,
        "EXPENSE TRACKER APP - DATA CLEANING & VALIDATION AUDIT REPORT",
        "=" * 70,
        f"Input Raw Dataset      : {input_csv_path}",
        f"Output Cleaned Dataset : {output_csv_path}",
        "",
        "1. ROW COUNT SUMMARY:",
        f"   - Initial Raw Rows     : {initial_rows}",
        f"   - Invalid Rows Dropped : {invalid_records_count}",
        f"   - Duplicates Removed   : {duplicates_removed}",
        f"   - Final Cleaned Rows   : {final_rows}",
        f"   - Retention Rate       : {(final_rows / initial_rows) * 100:.2f}%",
        "",
        "2. MISSING VALUES AUDIT (RAW DATA):",
        f"   - Total Missing Cells  : {total_initial_nulls}",
    ]

    for col, count in initial_nulls.items():
        report_lines.append(f"     * {col:<16} : {count} missing")

    report_lines.extend([
        "",
        "3. INVALID RECORDS BREAKDOWN & ACTIONS TAKEN:",
        f"   - Unparseable Dates    : {invalid_dates_count} (Action: Dropped invalid date records)",
        f"   - Non-numeric Amounts  : {invalid_amount_types_count} (Action: Dropped non-numeric values)",
        f"   - Non-positive Amounts : {non_positive_amount_count} (Action: Dropped amounts <= 0)",
        f"   - Invalid Types (!= Inc/Exp) : {invalid_type_count} (Action: Filtered strictly to Income/Expense)",
        "",
        "4. DUPLICATES AUDIT:",
        f"   - Duplicates Identified: {initial_duplicates}",
        f"   - Action Taken         : Kept first unique occurrence, removed identical copies.",
        "",
        "5. TEXT STANDARDIZATION:",
        "   - Category             : Stripped spaces, normalized to Title Case.",
        "   - Payment_Method       : Stripped spaces, normalized with 'UPI' capitalized.",
        "   - Type                 : Stripped spaces, capitalized ('Income' / 'Expense').",
        "   - Description          : Stripped whitespace.",
        "",
        "6. CLEANED DATASET METRICS:",
        f"   - Total Income Rows    : {int((df_clean['Type'] == 'Income').sum())}",
        f"   - Total Expense Rows   : {int((df_clean['Type'] == 'Expense').sum())}",
        f"   - Total Income Amount  : Rs. {df_clean[df_clean['Type'] == 'Income']['Amount'].sum():,.2f}",
        f"   - Total Expense Amount : Rs. {df_clean[df_clean['Type'] == 'Expense']['Amount'].sum():,.2f}",
        f"   - Date Span            : {df_clean['Date'].min()} to {df_clean['Date'].max()}",
        "",
        "=" * 70,
        "STATUS: DATA CLEANING PIPELINE COMPLETED SUCCESSFULLY",
        "=" * 70,
    ])

    report_content = "\n".join(report_lines)
    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"[+] Audit report saved to: '{output_report_path}'")
    print("=" * 70)

    return df_clean


def main():
    """
    Main entry point for command-line execution.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))

    raw_path = os.path.join(project_root, "data", "expenses.csv")
    cleaned_path = os.path.join(project_root, "data", "cleaned_expenses.csv")
    report_path = os.path.join(project_root, "outputs", "cleaning_report.txt")

    try:
        clean_expense_data(raw_path, cleaned_path, report_path)
    except Exception as err:
        print(f"[!] Data cleaning failed: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
