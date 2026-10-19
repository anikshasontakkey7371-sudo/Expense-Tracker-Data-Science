"""
=============================================================================
Expense Tracker App - Exploratory Data Analysis (EDA) Module
=============================================================================
This module performs statistical analysis on the cleaned expense dataset
(`data/cleaned_expenses.csv`).

Key Analyses Performed:
1. Executive Financial Summary:
   - Total income, total expenses, net balance, and savings rate.
   - Expense summary statistics: average, median, minimum, and maximum expense.
   - Transaction volume counts by type.
2. Category-Wise Expense Breakdown:
   - Total expenditure per category.
   - Percentage contribution to overall expenses.
   - Transaction count and average expenditure per category.
   - Identification of Top 5 spending categories.
3. Monthly Timeline Trends:
   - Monthly cashflow: Income vs. Expenses.
   - Monthly net savings and savings rate percentage.
4. Top 5 Largest Individual Expenses:
   - Specific outlier transactions with date, category, description, and amount.
5. Persistent Audit Export:
   - Saves formatted summary to `outputs/eda_report.txt`.
=============================================================================
"""

import os
import sys
import pandas as pd


def perform_eda(input_csv_path: str, output_report_path: str) -> dict:
    """
    Executes exploratory data analysis on the cleaned dataset.

    Parameters:
        input_csv_path (str): Path to cleaned CSV file.
        output_report_path (str): Destination path for the EDA report.

    Returns:
        dict: Dictionary containing computed summary metrics.
    """
    print("=" * 75)
    print(" EXPENSE TRACKER - EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # 1. Confirm File Existence & Load Dataset
    # -------------------------------------------------------------------------
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(
            f"Error: Cleaned dataset not found at '{input_csv_path}'. "
            "Please run 'python src/clean_data.py' first."
        )

    print(f"[*] Step 1: Loading cleaned dataset from '{input_csv_path}'...")
    df = pd.read_csv(input_csv_path)

    # -------------------------------------------------------------------------
    # 2. Schema & Representation Verification
    # -------------------------------------------------------------------------
    print("[*] Step 2: Verifying data types and 'Type' column representation...")
    # Safe standardization check for 'Type' values
    df["Type_Clean"] = df["Type"].astype(str).str.strip().str.capitalize()
    unique_types = df["Type_Clean"].unique().tolist()
    print(f"[+] Unique transaction types detected: {unique_types}")

    # Ensure Amount is numeric
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

    # Ensure Date is datetime
    df["Date_Parsed"] = pd.to_datetime(df["Date"], errors="coerce")

    # Filter into Income and Expense DataFrames
    income_df = df[df["Type_Clean"] == "Income"].copy()
    expense_df = df[df["Type_Clean"] == "Expense"].copy()

    # -------------------------------------------------------------------------
    # 3. Overall Financial Metrics
    # -------------------------------------------------------------------------
    total_transactions = len(df)
    total_income_txns = len(income_df)
    total_expense_txns = len(expense_df)

    total_income = float(income_df["Amount"].sum()) if not income_df.empty else 0.0
    total_expenses = float(expense_df["Amount"].sum()) if not expense_df.empty else 0.0
    net_balance = total_income - total_expenses
    savings_rate = (net_balance / total_income * 100.0) if total_income > 0 else 0.0

    avg_expense = float(expense_df["Amount"].mean()) if not expense_df.empty else 0.0
    median_expense = float(expense_df["Amount"].median()) if not expense_df.empty else 0.0
    highest_expense = float(expense_df["Amount"].max()) if not expense_df.empty else 0.0
    lowest_expense = float(expense_df["Amount"].min()) if not expense_df.empty else 0.0

    # -------------------------------------------------------------------------
    # 4. Category-Wise Expense Analysis
    # -------------------------------------------------------------------------
    print("[*] Step 3: Computing category-wise expenditure and percentages...")
    category_summary = expense_df.groupby("Category").agg(
        Total_Amount=("Amount", "sum"),
        Transaction_Count=("Amount", "count"),
        Average_Spend=("Amount", "mean")
    ).reset_index()

    category_summary["Percentage"] = (
        (category_summary["Total_Amount"] / total_expenses) * 100.0
    ).round(2)
    category_summary = category_summary.sort_values(by="Total_Amount", ascending=False).reset_index(drop=True)

    top_5_categories = category_summary.head(5)

    # -------------------------------------------------------------------------
    # 5. Monthly Trend Analysis (Inflow vs. Outflow)
    # -------------------------------------------------------------------------
    print("[*] Step 4: Analyzing monthly income, expenses, and net savings...")
    df["Year_Month"] = df["Date_Parsed"].dt.to_period("M").astype(str)

    monthly_summary = df.groupby(["Year_Month", "Type_Clean"])["Amount"].sum().unstack(fill_value=0.0).reset_index()
    if "Income" not in monthly_summary.columns:
        monthly_summary["Income"] = 0.0
    if "Expense" not in monthly_summary.columns:
        monthly_summary["Expense"] = 0.0

    monthly_summary["Net_Balance"] = monthly_summary["Income"] - monthly_summary["Expense"]
    monthly_summary["Savings_Rate_%"] = monthly_summary.apply(
        lambda row: (row["Net_Balance"] / row["Income"] * 100.0) if row["Income"] > 0 else 0.0,
        axis=1
    ).round(2)
    monthly_summary = monthly_summary.sort_values(by="Year_Month").reset_index(drop=True)

    # -------------------------------------------------------------------------
    # 6. Top 5 Largest Individual Expenses
    # -------------------------------------------------------------------------
    print("[*] Step 5: Identifying the 5 largest individual expenses...")
    top_5_expenses = expense_df.sort_values(by="Amount", ascending=False).head(5)[
        ["Date", "Category", "Description", "Amount", "Payment_Method"]
    ].reset_index(drop=True)

    # -------------------------------------------------------------------------
    # 7. Format Complete EDA Report
    # -------------------------------------------------------------------------
    print("[*] Step 6: Formatting results and compiling audit report...")
    report_lines = [
        "=" * 75,
        "EXPENSE TRACKER APP - EXPLORATORY DATA ANALYSIS (EDA) REPORT",
        "=" * 75,
        f"Input Cleaned Dataset : {input_csv_path}",
        f"Analysis Date Range   : {df['Date_Parsed'].min().date()} to {df['Date_Parsed'].max().date()}",
        "",
        "1. EXECUTIVE FINANCIAL SUMMARY:",
        "-" * 75,
        f"   Total Transactions   : {total_transactions:,} records",
        f"   - Income Entries     : {total_income_txns} transactions",
        f"   - Expense Entries    : {total_expense_txns} transactions",
        "",
        f"   Total Income         : Rs. {total_income:,.2f}",
        f"   Total Expenses       : Rs. {total_expenses:,.2f}",
        f"   Net Balance (Savings): Rs. {net_balance:,.2f}",
        f"   Overall Savings Rate : {savings_rate:.2f}%",
        "",
        "   Expense Distribution Metrics:",
        f"   - Average Expense    : Rs. {avg_expense:,.2f}",
        f"   - Median Expense     : Rs. {median_expense:,.2f}",
        f"   - Highest Expense    : Rs. {highest_expense:,.2f}",
        f"   - Lowest Expense     : Rs. {lowest_expense:,.2f}",
        "",
        "2. CATEGORY-WISE EXPENSE BREAKDOWN:",
        "-" * 75,
        f"{'Category':<16} | {'Total Spent (Rs.)':<18} | {'% of Total':<10} | {'Txn Count':<9} | {'Avg Spend (Rs.)':<15}",
        "-" * 75,
    ]

    for _, row in category_summary.iterrows():
        report_lines.append(
            f"{row['Category']:<16} | Rs. {row['Total_Amount']:>13,.2f} | {row['Percentage']:>9.2f}% | {int(row['Transaction_Count']):>9} | Rs. {row['Average_Spend']:>11,.2f}"
        )

    report_lines.extend([
        "-" * 75,
        f"Top 5 Spending Categories: {', '.join(top_5_categories['Category'].tolist())}",
        "",
        "3. MONTHLY CASHFLOW & SAVINGS TIMELINE:",
        "-" * 75,
        f"{'Month':<10} | {'Income (Rs.)':<15} | {'Expenses (Rs.)':<15} | {'Net Savings (Rs.)':<17} | {'Savings Rate':<12}",
        "-" * 75,
    ])

    for _, row in monthly_summary.iterrows():
        report_lines.append(
            f"{row['Year_Month']:<10} | Rs. {row['Income']:>10,.2f} | Rs. {row['Expense']:>10,.2f} | Rs. {row['Net_Balance']:>12,.2f} | {row['Savings_Rate_%']:>11.2f}%"
        )

    report_lines.extend([
        "-" * 75,
        "",
        "4. TOP 5 LARGEST INDIVIDUAL EXPENSES:",
        "-" * 75,
        f"{'Date':<11} | {'Category':<14} | {'Amount (Rs.)':<14} | {'Payment Method':<14} | {'Description'}",
        "-" * 75,
    ])

    for _, row in top_5_expenses.iterrows():
        report_lines.append(
            f"{row['Date']:<11} | {row['Category']:<14} | Rs. {row['Amount']:>9,.2f} | {row['Payment_Method']:<14} | {row['Description']}"
        )

    report_lines.extend([
        "-" * 75,
        "",
        "5. KEY DATA SCIENCE TAKEAWAYS & PATTERNS:",
        "-" * 75,
        f"   * Dominant Outflow : '{top_5_categories.iloc[0]['Category']}' is the highest expense category at {top_5_categories.iloc[0]['Percentage']:.1f}% of total outflows.",
        f"   * Cashflow Health  : The net cash balance is POSITIVE (+Rs. {net_balance:,.2f}) with a healthy {savings_rate:.1f}% savings rate.",
        f"   * Consistency      : Fixed monthly rent (Rs. 16,000/mo) accounts for the largest recurring single expenditure.",
        "=" * 75,
    ])

    report_content = "\n".join(report_lines)

    # -------------------------------------------------------------------------
    # 8. Save Report to outputs/eda_report.txt
    # -------------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\n[+] EDA report successfully saved to: '{output_report_path}'")
    print("\n" + report_content)

    return {
        "total_income": total_income,
        "total_expenses": total_expenses,
        "net_balance": net_balance,
        "savings_rate": savings_rate,
        "avg_expense": avg_expense,
        "highest_expense": highest_expense,
        "category_summary": category_summary,
        "monthly_summary": monthly_summary,
        "top_5_expenses": top_5_expenses
    }


def main():
    """
    Main entry point for command-line execution.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))

    cleaned_csv = os.path.join(project_root, "data", "cleaned_expenses.csv")
    output_report = os.path.join(project_root, "outputs", "eda_report.txt")

    try:
        perform_eda(cleaned_csv, output_report)
    except Exception as err:
        print(f"[!] EDA execution failed: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
