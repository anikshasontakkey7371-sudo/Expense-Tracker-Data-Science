"""
=============================================================================
Expense Tracker App - Automated Pipeline & Regression Test Suite
=============================================================================
This test suite verifies:
1. Data loading and missing file edge cases.
2. Handling of empty datasets, corrupted dates, and invalid/negative amounts.
3. Mathematical correctness of Income, Expense, and Net Balance calculations.
4. Filter simulation (Date range, Category, Payment Method, Type).
5. Chart generation safety with filtered data.
=============================================================================
"""

import os
import sys
import tempfile
import unittest
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.clean_data import clean_expense_data, standardize_payment_method
from src.analyze_data import perform_eda
from src.visualize_data import (
    plot_expenses_by_category_bar,
    plot_expenses_by_category_pie,
    plot_monthly_income_vs_expenses_line,
    plot_income_vs_expenses_bar
)


class TestExpenseTrackerPipeline(unittest.TestCase):
    """
    Comprehensive test cases covering all data science pipeline stages.
    """

    @classmethod
    def setUpClass(cls):
        cls.cleaned_csv_path = os.path.join(PROJECT_ROOT, "data", "cleaned_expenses.csv")
        cls.assertTrue(os.path.exists(cls.cleaned_csv_path), "Base cleaned_expenses.csv must exist.")
        cls.df = pd.read_csv(cls.cleaned_csv_path)

    # -------------------------------------------------------------------------
    # 1. Test Schema & Data Loading
    # -------------------------------------------------------------------------
    def test_schema_and_columns(self):
        """Verifies presence of all 6 required standardized columns."""
        required_cols = {"Date", "Category", "Description", "Amount", "Payment_Method", "Type"}
        self.assertTrue(required_cols.issubset(set(self.df.columns)))
        self.assertGreaterEqual(len(self.df), 200, "Dataset should have at least 200 rows.")

    def test_missing_file_handling(self):
        """Verifies FileNotFoundError is raised when an input CSV does not exist."""
        fake_path = os.path.join(PROJECT_ROOT, "data", "non_existent_file_xyz.csv")
        with tempfile.TemporaryDirectory() as temp_dir:
            out_csv = os.path.join(temp_dir, "out.csv")
            out_rep = os.path.join(temp_dir, "rep.txt")
            with self.assertRaises(FileNotFoundError):
                clean_expense_data(fake_path, out_csv, out_rep)

    # -------------------------------------------------------------------------
    # 2. Test Dirty Data & Edge Cases (Invalid Dates, Negative Amounts, Duplicates)
    # -------------------------------------------------------------------------
    def test_dirty_data_cleaning_and_validation(self):
        """
        Tests cleaning pipeline with dirty test data:
        - Corrupted date strings
        - Negative and zero amounts
        - Non-numeric amount strings
        - Text with irregular whitespace and casing
        - Duplicate rows
        """
        dirty_records = [
            # Valid Record
            {"Date": "2026-01-10", "Category": " food ", "Description": " lunch ", "Amount": 250.0, "Payment_Method": "upi", "Type": "expense"},
            # Duplicate of the above record (should be removed)
            {"Date": "2026-01-10", "Category": " food ", "Description": " lunch ", "Amount": 250.0, "Payment_Method": "upi", "Type": "expense"},
            # Invalid date (should be dropped)
            {"Date": "2026-99-99", "Category": "Travel", "Description": "Bus", "Amount": 50.0, "Payment_Method": "Cash", "Type": "Expense"},
            # Negative amount (should be dropped)
            {"Date": "2026-01-12", "Category": "Shopping", "Description": "Return item", "Amount": -500.0, "Payment_Method": "Credit Card", "Type": "Expense"},
            # Zero amount (should be dropped)
            {"Date": "2026-01-13", "Category": "Food", "Description": "Free sample", "Amount": 0.0, "Payment_Method": "Cash", "Type": "Expense"},
            # Non-numeric amount string (should be dropped)
            {"Date": "2026-01-14", "Category": "Rent", "Description": "Housing", "Amount": "not_a_number", "Payment_Method": "Net Banking", "Type": "Expense"},
            # Valid Income Record
            {"Date": "2026-01-01", "Category": "salary", "Description": "payroll", "Amount": 50000.0, "Payment_Method": "net banking", "Type": "income"},
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            dirty_csv = os.path.join(temp_dir, "dirty.csv")
            clean_csv = os.path.join(temp_dir, "clean.csv")
            report_txt = os.path.join(temp_dir, "report.txt")

            pd.DataFrame(dirty_records).to_csv(dirty_csv, index=False)

            cleaned_result = clean_expense_data(dirty_csv, clean_csv, report_txt)

            # Exactly 2 valid records should survive (1 Expense, 1 Income)
            self.assertEqual(len(cleaned_result), 2)
            self.assertEqual(set(cleaned_result["Type"]), {"Expense", "Income"})
            self.assertTrue(all(cleaned_result["Amount"] > 0))
            self.assertEqual(cleaned_result.loc[cleaned_result["Type"] == "Expense", "Category"].values[0], "Food")
            self.assertEqual(cleaned_result.loc[cleaned_result["Type"] == "Expense", "Payment_Method"].values[0], "UPI")
            self.assertTrue(os.path.exists(report_txt))

    # -------------------------------------------------------------------------
    # 3. Test Calculation Correctness (Income, Expenses, Net Balance)
    # -------------------------------------------------------------------------
    def test_calculation_correctness(self):
        """
        Verifies mathematical consistency:
        - Income sum strictly equals sum of rows where Type == 'Income'
        - Expense sum strictly equals sum of rows where Type == 'Expense'
        - Net Balance == Income - Expenses
        - No cross-contamination between Income and Expense
        """
        df = self.df.copy()
        df["Type"] = df["Type"].astype(str).str.strip().str.capitalize()
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

        income_mask = df["Type"] == "Income"
        expense_mask = df["Type"] == "Expense"

        # Check total counts
        self.assertEqual(len(df), income_mask.sum() + expense_mask.sum())

        expected_income = df.loc[income_mask, "Amount"].sum()
        expected_expense = df.loc[expense_mask, "Amount"].sum()
        expected_net = expected_income - expected_expense

        # Run EDA function to ensure internal logic matches
        with tempfile.TemporaryDirectory() as temp_dir:
            eda_rep = os.path.join(temp_dir, "eda_test.txt")
            metrics = perform_eda(self.cleaned_csv_path, eda_rep)

            self.assertAlmostEqual(metrics["total_income"], expected_income, places=2)
            self.assertAlmostEqual(metrics["total_expenses"], expected_expense, places=2)
            self.assertAlmostEqual(metrics["net_balance"], expected_net, places=2)

    # -------------------------------------------------------------------------
    # 4. Test Filter Simulations
    # -------------------------------------------------------------------------
    def test_filter_simulations(self):
        """
        Simulates Streamlit dashboard dynamic filtering behavior:
        - Category filter
        - Type filter
        - Date window filter
        """
        df = self.df.copy()
        df["Date"] = pd.to_datetime(df["Date"])

        # Filter by Category = ['Food']
        food_only = df[df["Category"] == "Food"]
        self.assertTrue(all(food_only["Category"] == "Food"))
        self.assertGreater(len(food_only), 0)

        # Filter by Type = ['Income']
        income_only = df[df["Type"] == "Income"]
        self.assertTrue(all(income_only["Type"] == "Income"))
        self.assertEqual((income_only["Type"] == "Expense").sum(), 0)

        # Filter by Date range (e.g. First 3 months)
        start_date = pd.to_datetime("2025-11-01")
        end_date = pd.to_datetime("2026-01-31")
        filtered_date = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
        self.assertTrue(all(filtered_date["Date"] >= start_date))
        self.assertTrue(all(filtered_date["Date"] <= end_date))
        self.assertLess(len(filtered_date), len(df))

    # -------------------------------------------------------------------------
    # 5. Test Chart Generation with Normal and Filtered Data
    # -------------------------------------------------------------------------
    def test_chart_functions_stability(self):
        """Verifies chart plotting functions execute and save valid PNGs."""
        df = self.df.copy()
        expense_df = df[df["Type"] == "Expense"].copy()

        with tempfile.TemporaryDirectory() as temp_dir:
            p1 = plot_expenses_by_category_bar(expense_df, temp_dir)
            p2 = plot_expenses_by_category_pie(expense_df, temp_dir)
            p3 = plot_monthly_income_vs_expenses_line(df, temp_dir)
            p4 = plot_income_vs_expenses_bar(df, temp_dir)

            self.assertTrue(os.path.exists(p1))
            self.assertTrue(os.path.exists(p2))
            self.assertTrue(os.path.exists(p3))
            self.assertTrue(os.path.exists(p4))
            self.assertGreater(os.path.getsize(p1), 1000)
            self.assertGreater(os.path.getsize(p2), 1000)
            self.assertGreater(os.path.getsize(p3), 1000)
            self.assertGreater(os.path.getsize(p4), 1000)


if __name__ == "__main__":
    unittest.main()
