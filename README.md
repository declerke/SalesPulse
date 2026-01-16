# 📈 SalesPulse: Multi-Source Sales Intelligence & Forecasting Engine

**SalesPulse** is a self-contained sales analytics solution designed to bridge the gap between raw, messy transactional data and professional business intelligence. It integrates real-world data from two distinct sources, performs rigorous time-series modeling, and delivers interactive insights through an executive-grade Power BI dashboard.



---

## 🎯 Project Goal
To provide a demonstrable, end-to-end solution that integrates multiple public data sources, performs rigorous data quality audits, and executes high-fidelity time-series forecasting (**ARIMA** & **Prophet**) to drive measurable business insights.

---

## 🧬 System Architecture
The pipeline is built on a modular "Ingest-Process-Predict-Visualize" logic:

1.  **Data Ingestion:** Merging **Global Superstore** (structured) and **UCI Online Retail** (messy/complex) datasets.
2.  **Quality Engine:** Automated audits for missing values, duplicates, and outlier detection using `pandas`.
3.  **Analytics Layer:** Feature engineering including date decomposition and seasonal trend analysis.
4.  **Forecasting Engine:** Multi-model approach using **ARIMA** for linear auto-regression and **Facebook Prophet** for complex seasonality.
5.  **BI Layer:** Interactive Power BI dashboard for executive reporting.



---

## 🛠️ Technical Stack
| Layer | Tools | Purpose |
| :--- | :--- | :--- |
| **Data Engineering** | Python 3.11, Pandas, NumPy | Cleaning, merging, and feature engineering |
| **Forecasting** | Prophet, Statsmodels (ARIMA) | Time-series analysis and future predictions |
| **Quality Audit** | Missingno, Great Expectations | Ensuring data integrity and schema validation |
| **Visualization** | Matplotlib, Seaborn | Static EDA and model evaluation plots |
| **Business Intelligence** | Power BI Desktop | Interactive dashboards and KPI tracking |

---

## 📊 Performance & Results
* **Data Integrity:** Successfully identified and resolved a 36-month data gap (2012-2014) that would have otherwise skewed growth metrics.
* **Stationarity:** Validated historical data using the **ADF (Augmented Dickey-Fuller) Test** ($p < 0.05$).
* **Forecasting Accuracy:** Optimized ARIMA (5,1,0) and Prophet models to minimize MAE (Mean Absolute Error).
* **Growth Metrics:** Identified an average **30.84%** Month-over-Month growth rate in the cleaned 2015-2018 period.



---

## 📂 Project Structure
```text
SalesPulse/
├── data/
│   ├── raw/                # Original Superstore & UCI CSVs
│   └── processed/          # Final cleaned_sales_final.csv & monthly_sales.csv
├── notebooks/
│   ├── 01_Ingestion_Quality_Audit.ipynb
│   ├── 02_Time_Series_Exploration.ipynb
│   └── 03_Forecasting_Models.ipynb
├── pbix/
│   └── SalesPulse_Executive_Dashboard.pbix
├── src/                    # Modular utility scripts (cleaning, plotting)
├── requirements.txt        # Python dependency list
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone & Environment
```bash
git clone [https://github.com/declerke/SalesPulse.git](https://github.com/declerke/SalesPulse.git)
cd SalesPulse
python -m venv .venv
# Windows: .venv\Scripts\activate | Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Pipeline
Execute the notebooks in the `/notebooks` directory sequentially to handle data merging, cleaning, and the generation of forecast artifacts.

### 3. View Dashboard
1. Open the `.pbix` file in **Power BI Desktop**.
2. Go to **Transform Data** > **Data Source Settings**.
3. Change Source to point to your local `data/processed/cleaned_sales_final.csv`.

---

## 🎓 Skills Demonstrated
* **End-to-End ETL:** Handling UTF-8 BOM encoding, mixed date-time strings, and temporal gaps.
* **Statistical Modeling:** Decomposing seasonality, trend, and noise in retail data.
* **Business Intelligence:** Crafting robust DAX measures for **YoY Growth**, **AOV**, and **Total Revenue**.
* **Problem Solving:** Addressing the "Dead Zone" gap between disparate datasets to maintain model validity.
