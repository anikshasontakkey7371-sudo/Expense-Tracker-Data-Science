# 💰 Expense Tracker App using Data Science

A comprehensive, placement-ready personal finance analytics project built with **Python**, **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**, and **Streamlit**.

---

## 📌 Project Overview
The **Expense Tracker App** helps users record, analyze, and visualize personal financial transactions. It demonstrates the complete end-to-end Data Science lifecycle:
1. **Synthetic Data Generation**: Reproducible synthetic financial transactions with realistic patterns.
2. **Data Cleaning & Schema Validation**: Robust pipelines handling missing values, edge cases, and data integrity.
3. **Exploratory Data Analysis (EDA)**: Statistical breakdowns of income, expenses, category spending, and monthly savings.
4. **Data Visualization**: High-resolution charts using Matplotlib and Seaborn.
5. **Interactive Streamlit Web Dashboard**: User-friendly analytics dashboard with dynamic date and category filtering.

---

## 📂 Project Structure

```text
Expense Tracker App/
│
├── data/
│   ├── generate_data.py       # Script to generate realistic synthetic transactions
│   └── expenses.csv           # Generated synthetic dataset
│
├── src/
│   └── __init__.py            # Python package initialization
│
├── outputs/                   # Exported charts and statistical reports
├── images/                    # UI screenshots and application visuals
├── notebooks/                 # Jupyter notebooks for exploratory data analysis
│
├── requirements.txt           # Project dependencies
└── README.md                  # Comprehensive project documentation
```

---

## 🛠️ Tech Stack & Dependencies

- **Language:** Python 3.10+
- **Data Manipulation:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Web App Framework:** `streamlit`

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Schema (`data/expenses.csv`)

The dataset contains the following standardized columns:

| Column | Data Type | Description | Sample Values |
| :--- | :--- | :--- | :--- |
| `Date` | `YYYY-MM-DD` | Date of the transaction | `2026-03-15`, `2026-08-01` |
| `Category` | `String` | Spending or earning category | `Food`, `Travel`, `Rent`, `Salary` |
| `Description` | `String` | Brief contextual detail | `Monthly corporate payroll salary`, `Grocery haul` |
| `Amount` | `Float` | Transaction amount in currency units (₹) | `16000.00`, `450.50` |
| `Payment_Method` | `String` | Channel of payment | `UPI`, `Credit Card`, `Debit Card`, `Net Banking`, `Cash` |
| `Type` | `String` | Direction of cash flow | `Income`, `Expense` |

---

## 🚀 How to Generate the Dataset

Run the data generator from your terminal:

```bash
python data/generate_data.py
```

### Key Highlights of the Generator:
- **Reproducibility:** Uses a fixed random seed (`seed=42`) so every run yields identical, verifiable records.
- **Realistic Cashflow:** Simulates fixed recurring transactions (monthly salary and rent) alongside everyday variable transactions (groceries, dining, fuel, shopping).
- **Scale:** Generates 300+ transactions spanning a 12-month timeline.

## 🚀 Quickstart Guide & Commands

### 1. Generate the Synthetic Dataset
```bash
python data/generate_data.py
```

### 2. Run Data Cleaning & Preprocessing
```bash
python src/clean_data.py
```

### 3. Run Exploratory Data Analysis (EDA)
```bash
python src/analyze_data.py
```

### 4. Generate Publication-Quality Visualizations
```bash
python -u src/visualize_data.py
```

### 5. Launch the Interactive Streamlit Web Dashboard
```bash
streamlit run app.py
```
*The web dashboard will automatically open in your default browser at `http://localhost:8501`.*

---

## 🛣️ Development Roadmap
- [x] **Phase 1:** Project directory structure, requirements, and synthetic dataset generation (`data/generate_data.py`).
- [x] **Phase 2:** Data cleaning pipeline, validation checks, and audit reporting (`src/clean_data.py`).
- [x] **Phase 3:** Exploratory Data Analysis, KPI calculations, and statistical reporting (`src/analyze_data.py`).
- [x] **Phase 4:** Publication-quality Matplotlib & Seaborn visualizations (`src/visualize_data.py`).
- [x] **Phase 5:** Interactive Streamlit web dashboard with dynamic filters and transaction logging (`app.py`).

