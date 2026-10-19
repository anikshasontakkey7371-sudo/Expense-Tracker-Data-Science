"""
=============================================================================
Expense Tracker App - Data Visualization Module (Phase 4)
=============================================================================
This module generates publication-quality financial charts using Pandas,
Matplotlib, and Seaborn.

Generated Charts:
1. Bar Chart: Total Expenses by Category (`outputs/charts/expenses_by_category_bar.png`)
2. Pie / Donut Chart: Percentage of Expenses by Category (`outputs/charts/expenses_by_category_pie.png`)
3. Line Chart: Monthly Income vs. Expenses Trend (`outputs/charts/monthly_income_vs_expenses_line.png`)
4. Bar Chart: Total Income vs. Total Expenses Comparison (`outputs/charts/income_vs_expenses_bar.png`)

All charts are saved at 300 DPI with clear labels, legends, and styling.
=============================================================================
"""

import os
import sys
import matplotlib
# Use 'Agg' non-interactive backend for headless environments & reliable server execution
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def setup_visual_style():
    """
    Sets up a modern, clean visual theme using Seaborn and Matplotlib.
    """
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


def plot_expenses_by_category_bar(expense_df: pd.DataFrame, output_dir: str) -> str:
    """
    Generates a horizontal bar chart displaying total expenses by category.
    """
    output_path = os.path.join(output_dir, "expenses_by_category_bar.png")

    if expense_df.empty:
        print("[!] Warning: Expense DataFrame is empty. Skipping category bar chart.")
        return ""

    # Group by category and sort descending
    cat_expenses = (
        expense_df.groupby("Category")["Amount"]
        .sum()
        .sort_values(ascending=True)  # Ascending for horizontal bar chart (highest at top)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    # Modern color palette
    palette = sns.color_palette("mako_r", n_colors=len(cat_expenses))
    bars = ax.barh(
        cat_expenses["Category"],
        cat_expenses["Amount"],
        color=palette,
        edgecolor="none",
        height=0.65
    )

    # Add numeric labels on each bar
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + (max(cat_expenses["Amount"]) * 0.015),
            bar.get_y() + bar.get_height() / 2,
            f"Rs. {width:,.0f}",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color="#2d3748"
        )

    ax.set_title("Total Expenses by Category", pad=15, color="#1a202c")
    ax.set_xlabel("Total Amount Spent (Rs.)", labelpad=10)
    ax.set_ylabel("Expense Category", labelpad=10)

    # Format x-axis with commas and room for labels
    ax.set_xlim(0, max(cat_expenses["Amount"]) * 1.18)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"Rs. {int(x):,}"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[+] Saved: {output_path}")
    return output_path


def plot_expenses_by_category_pie(expense_df: pd.DataFrame, output_dir: str) -> str:
    """
    Generates an elegant donut chart showing expense share percentages by category.
    """
    output_path = os.path.join(output_dir, "expenses_by_category_pie.png")

    if expense_df.empty:
        print("[!] Warning: Expense DataFrame is empty. Skipping category pie chart.")
        return ""

    cat_totals = (
        expense_df.groupby("Category")["Amount"]
        .sum()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(9, 8))

    # Harmonious curated palette
    colors = sns.color_palette("Set2", n_colors=len(cat_totals))

    wedges, texts, autotexts = ax.pie(
        cat_totals,
        labels=cat_totals.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        pctdistance=0.78,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2.5)  # Donut style
    )

    # Style label percentages
    for autotext in autotexts:
        autotext.set_color("#1a202c")
        autotext.set_fontsize(10)
        autotext.set_weight("bold")

    for text in texts:
        text.set_fontsize(10)
        text.set_weight("bold")
        text.set_color("#2d3748")

    total_expense = cat_totals.sum()
    # Add center text for total expenditure
    ax.text(
        0, 0,
        f"Total Spend\nRs. {total_expense:,.0f}",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#1a202c"
    )

    ax.set_title("Expense Distribution by Category (% Share)", pad=20, color="#1a202c")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[+] Saved: {output_path}")
    return output_path


def plot_monthly_income_vs_expenses_line(df: pd.DataFrame, output_dir: str) -> str:
    """
    Generates a dual-line trend chart comparing monthly Income vs. Expenses.
    """
    output_path = os.path.join(output_dir, "monthly_income_vs_expenses_line.png")

    # Ensure Date is parsed
    df = df.copy()
    df["Date_Parsed"] = pd.to_datetime(df["Date"], errors="coerce")
    valid_dates = df["Date_Parsed"].dropna()

    if len(valid_dates) < 2:
        print("[!] Warning: Insufficient date records. Skipping monthly line chart.")
        return ""

    df["Year_Month"] = df["Date_Parsed"].dt.to_period("M").astype(str)

    # Aggregate by Month and Type
    monthly = (
        df.groupby(["Year_Month", "Type"])["Amount"]
        .sum()
        .unstack(fill_value=0.0)
        .reset_index()
    )

    if "Income" not in monthly.columns:
        monthly["Income"] = 0.0
    if "Expense" not in monthly.columns:
        monthly["Expense"] = 0.0

    monthly = monthly.sort_values(by="Year_Month")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot Income line (Emerald Green)
    ax.plot(
        monthly["Year_Month"],
        monthly["Income"],
        marker="o",
        linewidth=2.8,
        markersize=7,
        color="#10b981",
        label="Monthly Income",
        alpha=0.9
    )

    # Plot Expense line (Coral Red)
    ax.plot(
        monthly["Year_Month"],
        monthly["Expense"],
        marker="s",
        linewidth=2.8,
        markersize=7,
        color="#ef4444",
        label="Monthly Expenses",
        alpha=0.9
    )

    # Highlight surplus area vs deficit area
    ax.fill_between(
        monthly["Year_Month"],
        monthly["Income"],
        monthly["Expense"],
        where=(monthly["Income"] >= monthly["Expense"]),
        color="#10b981",
        alpha=0.15,
        interpolate=True,
        label="Surplus (Savings)"
    )
    ax.fill_between(
        monthly["Year_Month"],
        monthly["Income"],
        monthly["Expense"],
        where=(monthly["Income"] < monthly["Expense"]),
        color="#ef4444",
        alpha=0.15,
        interpolate=True,
        label="Deficit (Overspend)"
    )

    ax.set_title("Monthly Cashflow: Income vs. Expenses Trend", pad=15, color="#1a202c")
    ax.set_xlabel("Billing Period (Year-Month)", labelpad=10)
    ax.set_ylabel("Amount (Rs.)", labelpad=10)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, loc: f"Rs. {int(y):,}"))
    plt.xticks(rotation=45, ha="right")

    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e1")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[+] Saved: {output_path}")
    return output_path


def plot_income_vs_expenses_bar(df: pd.DataFrame, output_dir: str) -> str:
    """
    Generates a clear side-by-side bar chart of Total Income vs. Total Expenses.
    """
    output_path = os.path.join(output_dir, "income_vs_expenses_bar.png")

    total_income = float(df[df["Type"] == "Income"]["Amount"].sum())
    total_expenses = float(df[df["Type"] == "Expense"]["Amount"].sum())
    net_savings = total_income - total_expenses

    fig, ax = plt.subplots(figsize=(8, 6))

    categories = ["Total Income", "Total Expenses", "Net Savings"]
    values = [total_income, total_expenses, net_savings]
    colors = ["#10b981", "#ef4444", "#3b82f6" if net_savings >= 0 else "#f59e0b"]

    bars = ax.bar(categories, values, color=colors, width=0.55, edgecolor="none")

    # Add numeric labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        y_pos = height + (max(values) * 0.02) if height >= 0 else height - (max(values) * 0.05)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            f"Rs. {height:,.0f}",
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=11,
            fontweight="bold",
            color="#1e293b"
        )

    ax.set_title("Overall Financial Balance: Income vs. Expenses", pad=15, color="#1a202c")
    ax.set_ylabel("Amount in Rupees (Rs.)", labelpad=10)

    # Reference zero line
    ax.axhline(0, color="#64748b", linewidth=1, linestyle="--")

    # Formatting limits and tick format
    ax.set_ylim(min(0, min(values) * 1.15), max(values) * 1.18)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, loc: f"Rs. {int(y):,}"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[+] Saved: {output_path}")
    return output_path


def generate_all_visualizations(
    cleaned_csv_path: str,
    output_charts_dir: str
) -> dict:
    """
    Coordinates the full chart generation pipeline.
    """
    print("=" * 70)
    print(" EXPENSE TRACKER - DATA VISUALIZATION PIPELINE")
    print("=" * 70)

    if not os.path.exists(cleaned_csv_path):
        raise FileNotFoundError(
            f"Error: Cleaned dataset not found at '{cleaned_csv_path}'. "
            "Please run 'python src/clean_data.py' first."
        )

    # Create charts directory if it does not exist
    os.makedirs(output_charts_dir, exist_ok=True)
    print(f"[*] Destination Directory: {output_charts_dir}")

    # Load dataset
    print(f"[*] Loading cleaned dataset from '{cleaned_csv_path}'...")
    df = pd.read_csv(cleaned_csv_path)

    # Standardize Type representation
    df["Type"] = df["Type"].astype(str).str.strip().str.capitalize()
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

    expense_df = df[df["Type"] == "Expense"].copy()
    income_df = df[df["Type"] == "Income"].copy()

    print(f"[+] Records loaded: {len(df)} total ({len(income_df)} Income, {len(expense_df)} Expense)")

    # Set visual theme
    setup_visual_style()

    # 1. Bar Chart: Total Expenses by Category
    print("[*] Generating Chart 1: Expenses by Category (Horizontal Bar)...")
    path_bar = plot_expenses_by_category_bar(expense_df, output_charts_dir)

    # 2. Pie / Donut Chart: Percentage of Expenses by Category
    print("[*] Generating Chart 2: Expenses by Category (% Share Donut Chart)...")
    path_pie = plot_expenses_by_category_pie(expense_df, output_charts_dir)

    # 3. Line Chart: Monthly Income vs Expenses Trend
    print("[*] Generating Chart 3: Monthly Income vs. Expenses Trend (Line Chart)...")
    path_line = plot_monthly_income_vs_expenses_line(df, output_charts_dir)

    # 4. Bar Chart: Income vs Expenses Summary
    print("[*] Generating Chart 4: Income vs. Expenses Summary (Bar Chart)...")
    path_summary = plot_income_vs_expenses_bar(df, output_charts_dir)

    print("=" * 70)
    print(" ALL CHARTS SUCCESSFULLY GENERATED AND SAVED")
    print("=" * 70)

    return {
        "expenses_by_category_bar": path_bar,
        "expenses_by_category_pie": path_pie,
        "monthly_income_vs_expenses_line": path_line,
        "income_vs_expenses_bar": path_summary
    }


def main():
    """
    Main entry point for command-line execution.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))

    cleaned_csv = os.path.join(project_root, "data", "cleaned_expenses.csv")
    charts_dir = os.path.join(project_root, "outputs", "charts")

    try:
        generate_all_visualizations(cleaned_csv, charts_dir)
    except Exception as err:
        print(f"[!] Chart generation failed: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
