"""
=============================================================================
Expense Tracker App - Synthetic Dataset Generator
=============================================================================
This script generates a realistic synthetic dataset for personal expense tracking.
It is built using Python, Pandas, and NumPy with beginner-friendly explanations.

Key Features:
- Fixed random seed (42) for 100% reproducible data generation.
- Generates 300+ transactions spanning a full 1-year timeline.
- Includes realistic recurring transactions (Monthly Salary & Rent).
- Follows the required schema:
    [Date, Category, Description, Amount, Payment_Method, Type]
- Saves the clean output as `data/expenses.csv`.
=============================================================================
"""

import os
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def set_seed(seed_value: int = 42) -> None:
    """
    Sets the random seed for both Python's random and NumPy's random modules.
    This ensures that every time you run this script, it generates the exact
    same dataset for reproducibility in tests and machine learning pipelines.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)


def generate_synthetic_expenses(num_daily_transactions: int = 300) -> pd.DataFrame:
    """
    Generates a realistic synthetic expense tracking dataset.

    Parameters:
        num_daily_transactions (int): Number of variable day-to-day transactions.

    Returns:
        pd.DataFrame: DataFrame containing all synthetic transactions.
    """
    # Fix the random seed for consistency
    set_seed(42)

    # Define the simulation time window (approx. past 12 months)
    start_date = datetime(2025, 11, 1)
    end_date = datetime(2026, 10, 15)
    total_days = (end_date - start_date).days

    # -------------------------------------------------------------------------
    # 1. Realistic Categories, Descriptions, and Payment Methods
    # -------------------------------------------------------------------------
    expense_catalog = {
        "Food": {
            "descriptions": [
                "Supermarket grocery haul",
                "Restaurant family dinner",
                "Morning cafe coffee & pastry",
                "Weekend food delivery order",
                "Fresh fruit & vegetable market",
                "Quick lunch at cafeteria",
                "Snacks and bakery items"
            ],
            "min_amount": 120.0,
            "max_amount": 1800.0,
            "payment_methods": ["UPI", "Credit Card", "Debit Card", "Cash"]
        },
        "Travel": {
            "descriptions": [
                "Metro card monthly recharge",
                "Uber / Ola city cab ride",
                "Petrol / Fuel refill",
                "Intercity express bus ticket",
                "Auto rickshaw fare",
                "Train journey ticket"
            ],
            "min_amount": 60.0,
            "max_amount": 1400.0,
            "payment_methods": ["UPI", "Credit Card", "Cash"]
        },
        "Shopping": {
            "descriptions": [
                "Online retail order (Amazon/Flipkart)",
                "Casual clothing & apparel",
                "Electronics & phone accessories",
                "Footwear & sneakers purchase",
                "Home decor and kitchenware",
                "Personal care & grooming products"
            ],
            "min_amount": 400.0,
            "max_amount": 6500.0,
            "payment_methods": ["Credit Card", "Debit Card", "UPI", "Net Banking"]
        },
        "Utilities": {
            "descriptions": [
                "Monthly electricity bill",
                "High-speed fiber internet bill",
                "Mobile prepaid/postpaid recharge",
                "Water utility municipal payment",
                "LPG cooking gas cylinder refill"
            ],
            "min_amount": 350.0,
            "max_amount": 2800.0,
            "payment_methods": ["UPI", "Net Banking", "Debit Card", "Credit Card"]
        },
        "Healthcare": {
            "descriptions": [
                "Pharmacy prescription medicines",
                "General doctor consultation fee",
                "Dental checkup & cleaning",
                "Multivitamins and health supplements",
                "Diagnostic lab blood test"
            ],
            "min_amount": 250.0,
            "max_amount": 3500.0,
            "payment_methods": ["UPI", "Debit Card", "Cash", "Credit Card"]
        },
        "Entertainment": {
            "descriptions": [
                "Weekend cinema multiplex movie tickets",
                "Streaming subscription (Netflix/Spotify)",
                "Video game purchase",
                "Live music concert pass",
                "Bowling and weekend arcade games"
            ],
            "min_amount": 199.0,
            "max_amount": 2200.0,
            "payment_methods": ["Credit Card", "UPI", "Debit Card"]
        }
    }

    records = []

    # -------------------------------------------------------------------------
    # 2. Add Recurring Monthly Transactions (Salary & Rent)
    # -------------------------------------------------------------------------
    # In real life, salaries arrive once a month (1st-3rd) and rent is paid (1st-5th).
    # Adding these explicitly creates realistic monthly cashflow patterns!
    current_month_start = datetime(2025, 11, 1)
    while current_month_start <= end_date:
        year = current_month_start.year
        month = current_month_start.month

        # --- Monthly Salary (Income) ---
        salary_day = random.randint(1, 3)
        salary_date = datetime(year, month, salary_day)
        salary_amount = float(random.choice([52000, 55000, 58000, 60000]))
        records.append({
            "Date": salary_date.strftime("%Y-%m-%d"),
            "Category": "Salary",
            "Description": "Monthly corporate payroll salary",
            "Amount": salary_amount,
            "Payment_Method": "Net Banking",
            "Type": "Income"
        })

        # --- Freelance / Bonus (Occasional Income) ---
        if random.random() < 0.40:  # 40% chance of freelance income per month
            freelance_day = random.randint(10, 25)
            freelance_date = datetime(year, month, freelance_day)
            records.append({
                "Date": freelance_date.strftime("%Y-%m-%d"),
                "Category": "Salary",  # Grouped under primary income / salary
                "Description": "Freelance web consulting milestone",
                "Amount": float(random.randint(8000, 18000)),
                "Payment_Method": "Net Banking",
                "Type": "Income"
            })

        # --- Monthly House Rent (Expense) ---
        rent_day = random.randint(1, 5)
        rent_date = datetime(year, month, rent_day)
        rent_amount = 16000.0
        records.append({
            "Date": rent_date.strftime("%Y-%m-%d"),
            "Category": "Rent",
            "Description": "Monthly apartment housing rent",
            "Amount": rent_amount,
            "Payment_Method": "Net Banking",
            "Type": "Expense"
        })

        # Move to the first day of next month
        if month == 12:
            current_month_start = datetime(year + 1, 1, 1)
        else:
            current_month_start = datetime(year, month + 1, 1)

    # -------------------------------------------------------------------------
    # 3. Add Variable Day-to-Day Expenses
    # -------------------------------------------------------------------------
    # Expense categories weights (Food & Travel occur more often than Healthcare)
    category_choices = ["Food", "Travel", "Shopping", "Utilities", "Healthcare", "Entertainment"]
    category_weights = [0.38, 0.22, 0.15, 0.10, 0.08, 0.07]

    for _ in range(num_daily_transactions):
        # Pick random day across the span
        random_day_offset = random.randint(0, total_days)
        txn_date = start_date + timedelta(days=random_day_offset)

        # Pick category based on realistic weights
        category = np.random.choice(category_choices, p=category_weights)
        cat_info = expense_catalog[category]

        # Pick random description and payment method from catalog
        description = random.choice(cat_info["descriptions"])
        payment_method = random.choice(cat_info["payment_methods"])

        # Pick amount within realistic bounds (rounded to 2 decimal places)
        raw_amount = np.random.uniform(cat_info["min_amount"], cat_info["max_amount"])
        # Round some to whole numbers or realistic endings
        amount = round(float(raw_amount), 2)

        records.append({
            "Date": txn_date.strftime("%Y-%m-%d"),
            "Category": category,
            "Description": description,
            "Amount": amount,
            "Payment_Method": payment_method,
            "Type": "Expense"
        })

    # -------------------------------------------------------------------------
    # 4. Create and Sort the DataFrame
    # -------------------------------------------------------------------------
    df = pd.DataFrame(records)

    # Sort chronologically by date
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by="Date").reset_index(drop=True)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    return df


def main():
    """
    Main entry point to generate and save synthetic dataset.
    """
    print("=" * 65)
    print(" EXPENSE TRACKER APP - SYNTHETIC DATASET GENERATOR")
    print("=" * 65)

    # Ensure output directory exists
    output_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(output_dir, ".."))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, "expenses.csv")

    print(f"[*] Generating synthetic transactions with fixed seed (42)...")
    # Generates ~300 daily expenses + ~25 monthly recurring items = ~325 total rows
    df = generate_synthetic_expenses(num_daily_transactions=300)

    # Save to CSV
    df.to_csv(csv_path, index=False)
    print(f"[+] Dataset successfully saved to: {csv_path}")

    # -------------------------------------------------------------------------
    # Display Verification & Summary
    # -------------------------------------------------------------------------
    print("\n" + "-" * 65)
    print(" DATASET VERIFICATION & SUMMARY")
    print("-" * 65)
    print(f"Total Rows Generated : {len(df)}")
    print(f"Columns Count        : {len(df.columns)}")
    print(f"Columns List         : {list(df.columns)}")
    print(f"Date Range           : {df['Date'].min()} to {df['Date'].max()}")
    print(f"Transaction Types    : {dict(df['Type'].value_counts())}")
    print(f"Categories Count     : {df['Category'].nunique()} unique categories")
    print(f"Categories Breakdown :\n{df['Category'].value_counts().to_string()}")

    income_total = df[df["Type"] == "Income"]["Amount"].sum()
    expense_total = df[df["Type"] == "Expense"]["Amount"].sum()
    savings = income_total - expense_total

    print("-" * 65)
    print(f"Total Synthetic Income  : Rs. {income_total:,.2f}")
    print(f"Total Synthetic Expenses: Rs. {expense_total:,.2f}")
    print(f"Net Synthetic Savings   : Rs. {savings:,.2f}")
    print("-" * 65)

    print("\nFirst 5 Sample Records:")
    print(df.head().to_string(index=False))
    print("=" * 65)


if __name__ == "__main__":
    main()
