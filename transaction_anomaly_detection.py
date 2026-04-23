"""
Transaction Anomaly Detection
==============================
Identifying suspicious patterns in financial transaction data using 
exploratory data analysis and rule-based anomaly detection.

Author: Affan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. GENERATE SYNTHETIC TRANSACTION DATASET
# ============================================================
# Simulating 50,000 transactions across 6 months for a mid-size 
# financial institution. Fraud is embedded at ~2.5% rate with 
# realistic patterns: off-hours spikes, amount anomalies, and 
# merchant category concentration.

np.random.seed(42)
n_transactions = 50000
n_customers = 3000

# Date range: 6 months
start_date = datetime(2025, 7, 1)
dates = [start_date + timedelta(
    days=np.random.randint(0, 180),
    hours=np.random.randint(0, 24),
    minutes=np.random.randint(0, 60)
) for _ in range(n_transactions)]

# Merchant categories
categories = [
    'Grocery', 'Restaurant', 'Online Retail', 'Fuel Station',
    'Electronics', 'Travel', 'Cash Withdrawal', 'Wire Transfer',
    'Luxury Goods', 'Gaming/Gambling'
]
cat_weights = [0.22, 0.18, 0.15, 0.12, 0.08, 0.07, 0.07, 0.04, 0.04, 0.03]

# Build base dataset
df = pd.DataFrame({
    'transaction_id': [f'TXN-{i:06d}' for i in range(n_transactions)],
    'customer_id': [f'CUST-{np.random.randint(1, n_customers+1):04d}' for _ in range(n_transactions)],
    'timestamp': sorted(dates),
    'category': np.random.choice(categories, n_transactions, p=cat_weights),
    'amount': np.round(np.abs(np.random.lognormal(mean=3.5, sigma=1.2, size=n_transactions)), 2)
})

df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.day_name()
df['month'] = df['timestamp'].dt.month

# Inject fraud patterns (~2.5%)
fraud_indices = np.random.choice(df.index, size=int(n_transactions * 0.025), replace=False)
df['is_fraud'] = 0
df.loc[fraud_indices, 'is_fraud'] = 1

# Fraudulent transactions: higher amounts, off-hours, concentrated categories
df.loc[fraud_indices, 'amount'] = np.round(
    np.abs(np.random.lognormal(mean=5.8, sigma=1.0, size=len(fraud_indices))), 2
)
df.loc[fraud_indices, 'hour'] = np.random.choice(
    [0, 1, 2, 3, 4, 23, 22], size=len(fraud_indices)
).astype(np.int32)
df.loc[fraud_indices, 'category'] = np.random.choice(
    ['Wire Transfer', 'Cash Withdrawal', 'Gaming/Gambling', 'Online Retail'],
    size=len(fraud_indices),
    p=[0.35, 0.30, 0.20, 0.15]
)

print(f"Dataset: {len(df):,} transactions | {df['customer_id'].nunique():,} customers")
print(f"Fraud rate: {df['is_fraud'].mean()*100:.1f}%")
print(f"Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
print(f"\n{df.head(10).to_string(index=False)}")

# Save raw data
df.to_csv('output/transactions_raw.csv', index=False)

# ============================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================

plt.style.use('seaborn-v0_8-whitegrid')
fig_params = dict(dpi=150, bbox_inches='tight')

# --- 2a. Transaction Amount Distribution: Fraud vs Legitimate ---

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (label, subset) in zip(axes, df.groupby('is_fraud')):
    name = 'Fraudulent' if label == 1 else 'Legitimate'
    color = '#e74c3c' if label == 1 else '#2ecc71'
    ax.hist(subset['amount'].clip(upper=2000), bins=60, color=color, alpha=0.8, edgecolor='white')
    ax.set_title(f'{name} Transactions (n={len(subset):,})', fontsize=13, fontweight='bold')
    ax.set_xlabel('Amount (capped at 2,000)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.axvline(subset['amount'].median(), color='black', linestyle='--', alpha=0.6,
               label=f"Median: {subset['amount'].median():,.0f}")
    ax.legend(fontsize=10)

fig.suptitle('Transaction Amount Distribution', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('output/01_amount_distribution.png', **fig_params)
plt.close()
print("\n[Saved] 01_amount_distribution.png")

# --- 2b. Fraud Concentration by Hour of Day ---

hourly = df.groupby(['hour', 'is_fraud']).size().unstack(fill_value=0)
hourly['fraud_rate'] = (hourly[1] / (hourly[0] + hourly[1])) * 100

fig, ax1 = plt.subplots(figsize=(12, 5))
bars = ax1.bar(hourly.index, hourly[0] + hourly[1], color='#3498db', alpha=0.4, label='Total Transactions')
ax2 = ax1.twinx()
ax2.plot(hourly.index, hourly['fraud_rate'], color='#e74c3c', linewidth=2.5, marker='o',
         markersize=6, label='Fraud Rate %')
ax2.fill_between(hourly.index, hourly['fraud_rate'], alpha=0.1, color='#e74c3c')

ax1.set_xlabel('Hour of Day', fontsize=12)
ax1.set_ylabel('Transaction Volume', fontsize=12, color='#3498db')
ax2.set_ylabel('Fraud Rate (%)', fontsize=12, color='#e74c3c')
ax1.set_xticks(range(24))

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

plt.title('Transaction Volume & Fraud Rate by Hour of Day', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('output/02_fraud_by_hour.png', **fig_params)
plt.close()
print("[Saved] 02_fraud_by_hour.png")

# --- 2c. Fraud Rate by Merchant Category ---

cat_fraud = df.groupby('category').agg(
    total=('is_fraud', 'count'),
    fraudulent=('is_fraud', 'sum'),
    avg_amount=('amount', 'mean')
).assign(fraud_rate=lambda x: (x['fraudulent'] / x['total']) * 100)
cat_fraud = cat_fraud.sort_values('fraud_rate', ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#e74c3c' if r > 5 else '#f39c12' if r > 2 else '#2ecc71' 
          for r in cat_fraud['fraud_rate']]
bars = ax.barh(cat_fraud.index, cat_fraud['fraud_rate'], color=colors, edgecolor='white', height=0.6)

for bar, val in zip(bars, cat_fraud['fraud_rate']):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')

ax.set_xlabel('Fraud Rate (%)', fontsize=12)
ax.set_title('Fraud Rate by Merchant Category', fontsize=14, fontweight='bold')
ax.axvline(x=df['is_fraud'].mean()*100, color='grey', linestyle='--', alpha=0.5, label='Overall Average')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('output/03_fraud_by_category.png', **fig_params)
plt.close()
print("[Saved] 03_fraud_by_category.png")

# --- 2d. Heatmap: Fraud Concentration by Hour x Day ---

pivot = df[df['is_fraud'] == 1].groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
pivot = pivot.reindex(day_order)

fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(pivot, cmap='YlOrRd', linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'Fraud Count'}, ax=ax)
ax.set_title('Fraud Concentration: Day of Week × Hour of Day', fontsize=14, fontweight='bold')
ax.set_xlabel('Hour of Day', fontsize=12)
ax.set_ylabel('', fontsize=12)
plt.tight_layout()
plt.savefig('output/04_fraud_heatmap.png', **fig_params)
plt.close()
print("[Saved] 04_fraud_heatmap.png")

# ============================================================
# 3. RULE-BASED ANOMALY DETECTION
# ============================================================
# Applying financial controls logic to flag suspicious transactions.
# Three rules derived from EDA findings:
#   Rule 1: Transaction amount > 97.5th percentile for its category
#   Rule 2: Transaction occurs between 11 PM and 5 AM
#   Rule 3: High-risk category (Wire Transfer, Cash Withdrawal, Gaming)

print("\n" + "="*60)
print("ANOMALY DETECTION: RULE-BASED FLAGGING")
print("="*60)

# Rule 1: Amount outlier within category
category_thresholds = df.groupby('category')['amount'].quantile(0.975)
df['amount_threshold'] = df['category'].map(category_thresholds)
df['flag_amount'] = (df['amount'] > df['amount_threshold']).astype(int)

# Rule 2: Off-hours transaction
df['flag_offhours'] = df['hour'].apply(lambda h: 1 if h >= 23 or h <= 4 else 0)

# Rule 3: High-risk category
high_risk = ['Wire Transfer', 'Cash Withdrawal', 'Gaming/Gambling']
df['flag_highrisk_cat'] = df['category'].apply(lambda c: 1 if c in high_risk else 0)

# Composite risk score (0-3)
df['risk_score'] = df['flag_amount'] + df['flag_offhours'] + df['flag_highrisk_cat']
df['flagged'] = (df['risk_score'] >= 2).astype(int)

# Evaluate detection performance
flagged_df = df[df['flagged'] == 1]
true_positives = flagged_df['is_fraud'].sum()
total_fraud = df['is_fraud'].sum()
total_flagged = len(flagged_df)

precision = true_positives / total_flagged * 100 if total_flagged > 0 else 0
recall = true_positives / total_fraud * 100 if total_fraud > 0 else 0

print(f"\nTotal transactions flagged: {total_flagged:,} ({total_flagged/len(df)*100:.1f}%)")
print(f"True fraudulent caught:     {true_positives:,} / {total_fraud:,}")
print(f"Precision:                  {precision:.1f}% (of flagged, how many were actually fraud)")
print(f"Recall:                     {recall:.1f}% (of all fraud, how many did we catch)")

# --- 3a. Detection Performance Visualization ---

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Confusion-style breakdown
labels = ['True Positive\n(Fraud caught)', 'False Positive\n(Legitimate flagged)',
          'False Negative\n(Fraud missed)', 'True Negative\n(Legitimate cleared)']
false_positives = total_flagged - true_positives
false_negatives = total_fraud - true_positives
true_negatives = len(df) - total_flagged - false_negatives
sizes = [true_positives, false_positives, false_negatives, true_negatives]
colors_pie = ['#2ecc71', '#f39c12', '#e74c3c', '#3498db']

axes[0].pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 9})
axes[0].set_title('Detection Breakdown', fontsize=13, fontweight='bold')

# Risk score distribution
risk_counts = df.groupby(['risk_score', 'is_fraud']).size().unstack(fill_value=0)
risk_counts.columns = ['Legitimate', 'Fraudulent']
risk_counts.plot(kind='bar', stacked=True, color=['#3498db', '#e74c3c'],
                 ax=axes[1], edgecolor='white')
axes[1].set_title('Risk Score Distribution', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Risk Score (0=Low, 3=High)', fontsize=11)
axes[1].set_ylabel('Transaction Count', fontsize=11)
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig('output/05_detection_performance.png', **fig_params)
plt.close()
print("[Saved] 05_detection_performance.png")

# ============================================================
# 4. HIGH-RISK CUSTOMER PROFILING
# ============================================================

print("\n" + "="*60)
print("HIGH-RISK CUSTOMER PROFILES")
print("="*60)

customer_risk = df.groupby('customer_id').agg(
    total_txns=('transaction_id', 'count'),
    total_amount=('amount', 'sum'),
    avg_amount=('amount', 'mean'),
    max_amount=('amount', 'max'),
    fraud_count=('is_fraud', 'sum'),
    avg_risk_score=('risk_score', 'mean'),
    flagged_count=('flagged', 'sum'),
    offhours_pct=('flag_offhours', 'mean')
).sort_values('avg_risk_score', ascending=False)

top_risk = customer_risk[customer_risk['flagged_count'] >= 2].head(15)
print(f"\nCustomers with 2+ flagged transactions: {len(customer_risk[customer_risk['flagged_count'] >= 2])}")
print(f"\nTop 15 Highest Risk Customers:")
print(top_risk[['total_txns', 'total_amount', 'fraud_count', 'flagged_count', 'avg_risk_score']].to_string())

# Save flagged transactions for review
flagged_export = df[df['flagged'] == 1][
    ['transaction_id', 'customer_id', 'timestamp', 'category', 'amount', 
     'risk_score', 'flag_amount', 'flag_offhours', 'flag_highrisk_cat', 'is_fraud']
].sort_values('risk_score', ascending=False)
flagged_export.to_csv('output/flagged_transactions.csv', index=False)

# ============================================================
# 5. SUMMARY
# ============================================================

print("\n" + "="*60)
print("KEY FINDINGS")
print("="*60)
print(f"""
1. TEMPORAL PATTERNS: Fraud concentrates heavily between 11 PM and 4 AM,
   with off-hours transactions showing {df[df['flag_offhours']==1]['is_fraud'].mean()*100:.0f}x the 
   baseline fraud rate.

2. CATEGORY RISK: Wire Transfers, Cash Withdrawals, and Gaming/Gambling 
   account for the majority of fraudulent activity despite representing 
   only {df[df['flag_highrisk_cat']==1].shape[0]/len(df)*100:.0f}% of total volume.

3. AMOUNT ANOMALIES: Fraudulent transactions average 
   {df[df['is_fraud']==1]['amount'].mean():,.0f} vs {df[df['is_fraud']==0]['amount'].mean():,.0f} 
   for legitimate ones ({df[df['is_fraud']==1]['amount'].mean()/df[df['is_fraud']==0]['amount'].mean():.1f}x higher).

4. DETECTION: A simple 3-rule system achieves {recall:.0f}% recall at 
   {precision:.0f}% precision, demonstrating that basic analytical rules 
   derived from pattern analysis can catch the majority of suspicious 
   activity before escalation to investigators.

Output files saved to /output directory.
""")
