"""
=============================================================================
Expense Tracker App - Interactive Streamlit Dashboard (Phase 5)
=============================================================================
A comprehensive, placement-ready personal finance dashboard built using
Streamlit, Pandas, NumPy, Matplotlib, and Seaborn.

Features:
- Dynamic Sidebar Filters: Date range, Category, Transaction Type, Payment Method.
- Executive KPI Cards: Total Income, Total Expenses, Net Balance, Total Transactions.
- Visual Charts:
  1. Expenses by Category (Horizontal Bar & Share Donut).
  2. Income vs. Expenses Comparison (Bar Chart).
  3. Monthly Cashflow & Savings Timeline (Dual Line Chart).
- Interactive Filtered Transaction Table with search, sorting, and CSV export.
- Interactive 'Add New Transaction' Form with live validation.
=============================================================================
"""

import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Expense Tracker | Data Science Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics and clean typography
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 18px 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-subtitle {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 4px;
    }
    .income-text { color: #10b981; }
    .expense-text { color: #ef4444; }
    .balance-positive { color: #0284c7; }
    .balance-negative { color: #f59e0b; }
    
    /* Tag Badges */
    .badge-income {
        background-color: #d1fae5;
        color: #065f46;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-expense {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Data Loading & Pipeline Integration
# -----------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cleaned_expenses.csv")
RAW_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "expenses.csv")


@st.cache_data
def load_data(csv_path: str):
    """
    Loads and prepares the cleaned expenses dataset.
    Cached with st.cache_data for optimal performance.
    """
    if not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return None

        # Ensure correct types
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
        df["Category"] = df["Category"].astype(str).str.strip().str.title()
        df["Type"] = df["Type"].astype(str).str.strip().str.capitalize()
        df["Payment_Method"] = df["Payment_Method"].astype(str).str.strip()

        # Drop any nulls created by bad records
        df = df.dropna(subset=["Date", "Amount", "Type"]).copy()
        df.sort_values(by="Date", ascending=False, inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None


# -----------------------------------------------------------------------------
# 3. Application Header & Dataset Status
# -----------------------------------------------------------------------------
st.title("💰 Expense Tracker & Financial Analytics")
st.caption("A Data Science Portfolio Project powered by Python, Pandas, Matplotlib, Seaborn, and Streamlit")

df_raw = load_data(DATA_PATH)

# Handle Missing or Empty Dataset Gracefully
if df_raw is None or df_raw.empty:
    st.error("⚠️ Cleaned dataset not found or empty at `data/cleaned_expenses.csv`.")
    st.info(
        "To get started, please generate the synthetic dataset and run the cleaning pipeline:\n"
        "1. Run `python data/generate_data.py`\n"
        "2. Run `python src/clean_data.py`"
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        if st.button("⚡ Run Pipeline Automatically"):
            with st.spinner("Generating and cleaning data..."):
                try:
                    import subprocess
                    subprocess.run([sys.executable, "data/generate_data.py"], check=True)
                    subprocess.run([sys.executable, "src/clean_data.py"], check=True)
                    st.success("Dataset successfully generated and cleaned! Please refresh.")
                    st.rerun()
                except Exception as err:
                    st.error(f"Pipeline execution failed: {err}")
    st.stop()


# -----------------------------------------------------------------------------
# 4. Sidebar Dynamic Filters
# -----------------------------------------------------------------------------
st.sidebar.header("🔍 Filter Transactions")

# 1. Date Range Filter
min_date = df_raw["Date"].min().date()
max_date = df_raw["Date"].max().date()

date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    help="Filter transactions within a specific start and end date."
)

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

# 2. Transaction Type Filter
available_types = sorted(df_raw["Type"].unique().tolist())
selected_types = st.sidebar.multiselect(
    "Transaction Type",
    options=available_types,
    default=available_types,
    help="Select Income, Expense, or both."
)

# 3. Category Filter
available_categories = sorted(df_raw["Category"].unique().tolist())
selected_categories = st.sidebar.multiselect(
    "Category",
    options=available_categories,
    default=available_categories,
    help="Select one or more spending/income categories."
)

# 4. Payment Method Filter
available_methods = sorted(df_raw["Payment_Method"].unique().tolist())
selected_methods = st.sidebar.multiselect(
    "Payment Method",
    options=available_methods,
    default=available_methods,
    help="Filter by payment channel (e.g. UPI, Net Banking, Credit Card)."
)

# Reset Button
if st.sidebar.button("🔄 Reset All Filters"):
    st.rerun()

# -----------------------------------------------------------------------------
# 5. Apply Filters
# -----------------------------------------------------------------------------
filtered_df = df_raw[
    (df_raw["Date"].dt.date >= start_date) &
    (df_raw["Date"].dt.date <= end_date) &
    (df_raw["Type"].isin(selected_types)) &
    (df_raw["Category"].isin(selected_categories)) &
    (df_raw["Payment_Method"].isin(selected_methods))
].copy()

# Filter Status Display
st.sidebar.markdown("---")
st.sidebar.caption(f"Showing **{len(filtered_df):,}** of **{len(df_raw):,}** transactions")


# Check if filtered data is empty
if filtered_df.empty:
    st.warning("⚠️ No transactions match your selected filter criteria. Please broaden your filters in the sidebar.")
    st.stop()


# -----------------------------------------------------------------------------
# 6. Executive KPI Cards
# -----------------------------------------------------------------------------
income_subset = filtered_df[filtered_df["Type"] == "Income"]
expense_subset = filtered_df[filtered_df["Type"] == "Expense"]

total_income = float(income_subset["Amount"].sum())
total_expenses = float(expense_subset["Amount"].sum())
net_balance = total_income - total_expenses
total_txns = len(filtered_df)
savings_rate = (net_balance / total_income * 100.0) if total_income > 0 else 0.0

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Income</div>
        <div class="metric-value income-text">Rs. {total_income:,.2f}</div>
        <div class="metric-subtitle">{len(income_subset)} income transactions</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Expenses</div>
        <div class="metric-value expense-text">Rs. {total_expenses:,.2f}</div>
        <div class="metric-subtitle">{len(expense_subset)} expense transactions</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    balance_class = "balance-positive" if net_balance >= 0 else "balance-negative"
    sign = "+" if net_balance > 0 else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Net Balance</div>
        <div class="metric-value {balance_class}">{sign}Rs. {net_balance:,.2f}</div>
        <div class="metric-subtitle">Savings Rate: {savings_rate:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    avg_expense = expense_subset["Amount"].mean() if not expense_subset.empty else 0.0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Transactions</div>
        <div class="metric-value">{total_txns:,}</div>
        <div class="metric-subtitle">Avg Expense: Rs. {avg_expense:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 7. Analytics & Visualization Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Visual Analytics",
    "📈 Monthly Trends",
    "📋 Transaction Records",
    "➕ Add New Transaction"
])

# Set consistent Matplotlib style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9
})

# -----------------------------------------------------------------------------
# TAB 1: Visual Analytics (Category & Comparison)
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Spending Analysis & Distribution")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("**Expenses by Category (Total Amount)**")
        if not expense_subset.empty:
            cat_totals = (
                expense_subset.groupby("Category")["Amount"]
                .sum()
                .sort_values(ascending=True)
                .reset_index()
            )

            fig, ax = plt.subplots(figsize=(7, 4.5))
            palette = sns.color_palette("mako_r", n_colors=len(cat_totals))
            bars = ax.barh(cat_totals["Category"], cat_totals["Amount"], color=palette, height=0.6)

            for bar in bars:
                width = bar.get_width()
                ax.text(
                    width + (max(cat_totals["Amount"]) * 0.02),
                    bar.get_y() + bar.get_height() / 2,
                    f"Rs. {width:,.0f}",
                    va="center", ha="left", fontsize=8.5, fontweight="bold", color="#1e293b"
                )

            ax.set_xlabel("Total Amount (Rs.)", labelpad=8)
            ax.set_ylabel("")
            ax.set_xlim(0, max(cat_totals["Amount"]) * 1.25)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"Rs. {int(x):,}"))
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No expense transactions in the selected filter range.")

    with col_chart2:
        st.markdown("**Expense Share by Category (% Share)**")
        if not expense_subset.empty:
            cat_pie = (
                expense_subset.groupby("Category")["Amount"]
                .sum()
                .sort_values(ascending=False)
            )

            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            colors = sns.color_palette("Set2", n_colors=len(cat_pie))
            wedges, texts, autotexts = ax.pie(
                cat_pie,
                labels=cat_pie.index,
                autopct="%1.1f%%",
                startangle=140,
                colors=colors,
                pctdistance=0.75,
                wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2)
            )
            for at in autotexts:
                at.set_fontsize(8.5)
                at.set_weight("bold")
            for t in texts:
                t.set_fontsize(9)

            ax.text(
                0, 0, f"Total\nRs. {total_expenses:,.0f}",
                ha="center", va="center", fontsize=10, fontweight="bold", color="#0f172a"
            )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No expense transactions in the selected filter range.")

    st.markdown("---")

    # Overall Inflow vs Outflow Comparison Bar
    st.markdown("**Macro Cashflow Comparison: Total Income vs. Total Expenses**")
    fig_comp, ax_comp = plt.subplots(figsize=(8, 3.5))
    cats = ["Total Income", "Total Expenses", "Net Savings"]
    vals = [total_income, total_expenses, net_balance]
    cols = ["#10b981", "#ef4444", "#0284c7" if net_balance >= 0 else "#f59e0b"]

    bars_comp = ax_comp.bar(cats, vals, color=cols, width=0.45)
    for b in bars_comp:
        h = b.get_height()
        y_loc = h + (max(vals) * 0.03) if h >= 0 else h - (max(vals) * 0.06)
        ax_comp.text(
            b.get_x() + b.get_width() / 2, y_loc,
            f"Rs. {h:,.2f}",
            ha="center", va="bottom" if h >= 0 else "top",
            fontsize=9.5, fontweight="bold", color="#1e293b"
        )

    ax_comp.axhline(0, color="#94a3b8", linestyle="--", linewidth=1)
    ax_comp.set_ylabel("Amount (Rs.)")
    ax_comp.set_ylim(min(0, min(vals) * 1.2), max(vals) * 1.25)
    ax_comp.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f"Rs. {int(y):,}"))
    plt.tight_layout()
    st.pyplot(fig_comp)
    plt.close(fig_comp)


# -----------------------------------------------------------------------------
# TAB 2: Monthly Timeline Trends
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Monthly Income vs. Expenses Timeline")

    df_monthly = filtered_df.copy()
    df_monthly["Year_Month"] = df_monthly["Date"].dt.to_period("M").astype(str)

    monthly_agg = (
        df_monthly.groupby(["Year_Month", "Type"])["Amount"]
        .sum()
        .unstack(fill_value=0.0)
        .reset_index()
    )

    if "Income" not in monthly_agg.columns:
        monthly_agg["Income"] = 0.0
    if "Expense" not in monthly_agg.columns:
        monthly_agg["Expense"] = 0.0

    monthly_agg["Net_Savings"] = monthly_agg["Income"] - monthly_agg["Expense"]
    monthly_agg = monthly_agg.sort_values(by="Year_Month")

    if len(monthly_agg) > 1:
        fig_line, ax_line = plt.subplots(figsize=(10, 4.5))

        ax_line.plot(
            monthly_agg["Year_Month"], monthly_agg["Income"],
            marker="o", linewidth=2.5, color="#10b981", label="Income"
        )
        ax_line.plot(
            monthly_agg["Year_Month"], monthly_agg["Expense"],
            marker="s", linewidth=2.5, color="#ef4444", label="Expenses"
        )

        ax_line.fill_between(
            monthly_agg["Year_Month"],
            monthly_agg["Income"],
            monthly_agg["Expense"],
            where=(monthly_agg["Income"] >= monthly_agg["Expense"]),
            color="#10b981", alpha=0.15, label="Surplus (Savings)"
        )
        ax_line.fill_between(
            monthly_agg["Year_Month"],
            monthly_agg["Income"],
            monthly_agg["Expense"],
            where=(monthly_agg["Income"] < monthly_agg["Expense"]),
            color="#ef4444", alpha=0.15, label="Deficit (Overspend)"
        )

        ax_line.set_xlabel("Billing Period")
        ax_line.set_ylabel("Amount (Rs.)")
        ax_line.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f"Rs. {int(y):,}"))
        plt.xticks(rotation=45, ha="right")
        ax_line.legend(loc="upper right", frameon=True, facecolor="white")
        plt.tight_layout()
        st.pyplot(fig_line)
        plt.close(fig_line)

        # Monthly Table Breakdown
        st.markdown("**Monthly Cashflow Summary Table**")
        display_monthly = monthly_agg.copy()
        display_monthly["Income"] = display_monthly["Income"].apply(lambda v: f"Rs. {v:,.2f}")
        display_monthly["Expense"] = display_monthly["Expense"].apply(lambda v: f"Rs. {v:,.2f}")
        display_monthly["Net_Savings"] = display_monthly["Net_Savings"].apply(lambda v: f"Rs. {v:,.2f}")
        st.dataframe(display_monthly, use_container_width=True, hide_index=True)
    else:
        st.info("Select a date range spanning at least 2 months to view timeline trends.")


# -----------------------------------------------------------------------------
# TAB 3: Transaction Records & Search
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Transaction History & Search")

    col_search, col_sort = st.columns([3, 2])
    with col_search:
        search_query = st.text_input("🔍 Search by Description or Category", placeholder="e.g. Grocery, Rent, Salary...")
    with col_sort:
        sort_by = st.selectbox(
            "Sort Records By",
            options=[
                "Date (Newest First)",
                "Date (Oldest First)",
                "Amount (Highest First)",
                "Amount (Lowest First)"
            ]
        )

    table_df = filtered_df.copy()

    # Apply search filter
    if search_query:
        query = search_query.lower()
        table_df = table_df[
            table_df["Description"].str.lower().str.contains(query, na=False) |
            table_df["Category"].str.lower().str.contains(query, na=False)
        ]

    # Apply sorting
    if sort_by == "Date (Newest First)":
        table_df.sort_values(by="Date", ascending=False, inplace=True)
    elif sort_by == "Date (Oldest First)":
        table_df.sort_values(by="Date", ascending=True, inplace=True)
    elif sort_by == "Amount (Highest First)":
        table_df.sort_values(by="Amount", ascending=False, inplace=True)
    elif sort_by == "Amount (Lowest First)":
        table_df.sort_values(by="Amount", ascending=True, inplace=True)

    # Format table for display
    display_table = table_df.copy()
    display_table["Date"] = display_table["Date"].dt.strftime("%Y-%m-%d")
    display_table["Amount"] = display_table["Amount"].apply(lambda x: f"Rs. {x:,.2f}")

    st.dataframe(
        display_table[["Date", "Category", "Description", "Amount", "Payment_Method", "Type"]],
        use_container_width=True,
        hide_index=True
    )

    # Download CSV Button
    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered Transactions as CSV",
        data=csv_bytes,
        file_name="filtered_expenses.csv",
        mime="text/csv"
    )


# -----------------------------------------------------------------------------
# TAB 4: Add New Transaction
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("Log a New Transaction")
    st.caption("Add an income or expense transaction with automatic schema validation.")

    with st.form("add_txn_form", clear_on_submit=True):
        col_f1, col_f2 = st.columns(2)

        with col_f1:
            new_date = st.date_input("Transaction Date", value=datetime.today())
            new_type = st.selectbox("Transaction Type", options=["Expense", "Income"])
            new_category = st.selectbox(
                "Category",
                options=[
                    "Food", "Travel", "Shopping", "Utilities", "Healthcare",
                    "Entertainment", "Rent", "Salary", "Freelance", "Other"
                ]
            )

        with col_f2:
            new_amount = st.number_input("Amount (Rs.)", min_value=1.0, value=500.0, step=50.0)
            new_method = st.selectbox(
                "Payment Method",
                options=["UPI", "Credit Card", "Debit Card", "Net Banking", "Cash"]
            )
            new_desc = st.text_input("Description", placeholder="e.g. Supermarket Grocery Run")

        submit_btn = st.form_submit_button("💾 Save Transaction")

        if submit_btn:
            if not new_desc.strip():
                st.error("Please enter a valid description for the transaction.")
            elif new_amount <= 0:
                st.error("Amount must be greater than zero.")
            else:
                new_record = {
                    "Date": new_date.strftime("%Y-%m-%d"),
                    "Category": new_category.title(),
                    "Description": new_desc.strip(),
                    "Amount": round(float(new_amount), 2),
                    "Payment_Method": new_method,
                    "Type": new_type.capitalize()
                }

                # Append to cleaned_expenses.csv
                new_row_df = pd.DataFrame([new_record])
                new_row_df.to_csv(DATA_PATH, mode="a", header=False, index=False)

                # Clear cache and reload
                load_data.clear()
                st.success(f"Successfully recorded {new_type}: Rs. {new_amount:,.2f} for '{new_desc}'!")
                st.rerun()


# -----------------------------------------------------------------------------
# 8. Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94a3b8; font-size: 0.85rem;'>"
    "Expense Tracker App | Placement-Ready Data Science Project | Built with Streamlit, Pandas, Matplotlib & Seaborn"
    "</div>",
    unsafe_allow_html=True
)
