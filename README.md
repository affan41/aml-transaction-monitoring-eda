# Transaction Anomaly Detection

Exploratory data analysis and rule-based anomaly detection on financial transaction data, identifying suspicious patterns through temporal analysis, merchant category risk profiling, and composite scoring.

## Objective

Financial institutions process millions of transactions daily, and manual review of every transaction is not feasible. This project applies data analysis techniques to identify potentially fraudulent transactions by examining patterns in timing, amounts, and merchant categories — translating financial controls knowledge into a data-driven detection framework.

## Approach

1. **Exploratory Data Analysis** — Profiling transaction distributions, identifying temporal fraud concentration (off-hours spikes), and mapping fraud rates across merchant categories.

2. **Rule-Based Anomaly Detection** — Three detection rules derived from EDA findings:
   - Amount outliers exceeding the 97.5th percentile within each merchant category
   - Off-hours transactions (11 PM – 5 AM)
   - High-risk merchant categories (Wire Transfers, Cash Withdrawals, Gaming/Gambling)

3. **Composite Risk Scoring** — Transactions flagged on 2+ rules are escalated, achieving meaningful recall while keeping false positives manageable.

4. **Customer Risk Profiling** — Aggregating transaction-level flags to identify high-risk customer accounts for enhanced due diligence.

## Key Findings

- Fraudulent transactions concentrate between 11 PM and 4 AM, with off-hours activity showing significantly elevated fraud rates
- Wire Transfers, Cash Withdrawals, and Gaming/Gambling represent a disproportionate share of fraud despite low overall volume
- A simple 3-rule composite system demonstrates that pattern-based detection can catch the majority of suspicious activity before escalation

## Visualisations

| Output | Description |
|--------|-------------|
| `01_amount_distribution.png` | Transaction amount distributions — fraud vs legitimate |
| `02_fraud_by_hour.png` | Dual-axis: transaction volume and fraud rate by hour |
| `03_fraud_by_category.png` | Fraud rate by merchant category with risk thresholds |
| `04_fraud_heatmap.png` | Fraud concentration heatmap: day of week × hour |
| `05_detection_performance.png` | Detection breakdown and risk score distribution |

## Tech Stack

- Python 3
- pandas, NumPy — data manipulation
- matplotlib, seaborn — visualisation

## How to Run

```bash
pip install pandas numpy matplotlib seaborn
python transaction_anomaly_detection.py
```

Output charts and flagged transaction CSV are saved to the `output/` directory.

## Background

This project was built as preparatory work for the MSc Business Analytics programme, applying existing financial controls knowledge (ACCA qualification) through analytical and programming tools. The approach mirrors real-world AML transaction monitoring workflows where rule-based systems serve as a first line of detection before machine learning models are layered on top.
