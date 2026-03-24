# 🏦 Customer Delinquency Prediction — Tata Group Gen AI Job Simulation

> Predictive model to identify customers at risk of account delinquency — built as part of the **Tata Group Gen AI Data Analysis Job Simulation** on Forage.

---

## 📌 Project Overview

**Geldium Finance** needed an early warning system to predict which customers were likely to miss payments and become delinquent. This project delivers an end-to-end machine learning pipeline — from raw data exploration to a fairness-assessed, deployment-ready Logistic Regression model.

GenAI tools (DeepSeek, ChatGPT) were used throughout to assist with EDA summarization, imputation strategy design, model scaffolding, and evaluation planning — mirroring real-world AI-augmented data workflows.

---

## 🗂️ Repository Structure

```
tata-genai-delinquency-prediction/     # Full pipeline script
├── tata_delinquency_model.ipynb    # Jupyter notebook (with outputs)
├── tata_model_dashboard.png        # Evaluation dashboard (6 charts)
└── README.md                       # You are here
```

---

## 🧩 Tasks Completed

### Task 1 — Exploratory Data Analysis (EDA)
- Audited a 500-record financial dataset with 18+ features
- Identified and documented **3 sources of missing data** (Income: 5.4%, Loan_Balance: 4.4%, Credit_Score: 0.2%)
- Detected data quality issues: `Employment_Status` had 6 inconsistent formats; `Credit_Utilization` contained impossible values (> 1.0)
- Uncovered key risk indicators through correlation analysis and group-level delinquency rates
- Used GenAI prompts to accelerate pattern detection and imputation strategy selection

### Task 2 — Predictive Modeling
- Designed and implemented a full ML pipeline: cleaning → feature engineering → training → evaluation → fairness assessment
- Chose **Logistic Regression** for interpretability and regulatory compliance (explainable by design)
- Engineered features from 6 months of payment history (`Payment_Issues`, `Recent_Trend`)
- Optimized classification threshold using Precision-Recall curve (default 0.5 → optimal 0.608)
- Conducted fairness audit across Employment Status, Location, and Credit Card Type

---

## ⚙️ Pipeline at a Glance

```
Raw Data (500 records)
    │
    ▼
EDA & Quality Audit
    │   • Missing value detection
    │   • Anomaly flagging
    │   • Correlation analysis
    ▼
Data Cleaning
    │   • Standardize Employment_Status (6 → 5 categories)
    │   • Cap Credit_Utilization at 1.0
    │   • Median imputation (grouped by employment & credit bin)
    ▼
Feature Engineering
    │   • Payment_Issues (sum of Late + Missed across 6 months)
    │   • Recent_Trend (issues in last 3 months)
    │   • Imputation flags (Income_Imputed, Loan_Balance_Imputed)
    ▼
Model Training
    │   • Logistic Regression (class_weight='balanced')
    │   • 75/25 stratified train-test split
    │   • StandardScaler on numerical features
    │   • One-hot encoding for categoricals
    ▼
Evaluation & Threshold Tuning
    │   • ROC-AUC, F1, Precision, Recall, Brier Score
    │   • 5-fold Stratified Cross-Validation
    │   • Optimal threshold via Precision-Recall curve
    ▼
Fairness Assessment
        • Disparate Impact Ratio across demographic groups
        • True Positive Rate parity check
```

---

## 📊 Model Performance

| Metric | Score | Target |
|--------|-------|--------|
| ROC-AUC | **0.794** | > 0.80 |
| Recall (Delinquent) | **0.85** | > 0.75 ✅ |
| F1 Score (optimised) | **0.49** | > 0.72 |
| Brier Score | 0.184 | < 0.15 |
| Accuracy | **85%** (optimal threshold) | > 0.80 ✅ |

> ⚠️ **Why Logistic Regression?** In financial services, regulatory frameworks (like RBI guidelines in India) require models to be *explainable*. A model that achieves 95% AUC but can't explain its decisions is a compliance risk. Logistic Regression's coefficients directly map to business logic: *"5+ missed payments increases delinquency odds by ~4x."*

---

## 🔍 Top Risk Factors Identified

1. **Missed_Payments** — Strongest direct predictor; 4+ missed payments → 33–50% delinquency rate
2. **Debt_to_Income_Ratio** — Ratios > 0.4 strongly correlated with delinquency
3. **Credit_Utilization** — Utilization > 0.8 appears in 40% of delinquent accounts
4. **Employment_Status** — Unemployed customers show 2.5x higher delinquency vs employed
5. **Recent_Trend** — Payment deterioration in the last 3 months is a leading indicator

---

## ⚖️ Fairness & Bias Assessment

The model was evaluated for **disparate impact** across three demographic dimensions:

| Group | Disparate Impact Ratio | Status |
|-------|------------------------|--------|
| Employment Status | 0.22 | ⚠️ Requires review |
| Location | 0.13 | ⚠️ Requires review |
| Credit Card Type | 0.12 | ⚠️ Requires review |

> Disparate impact < 0.80 signals that prediction rates differ significantly across groups. Mitigation steps recommended: re-weighted training, threshold adjustment per group, and removal of proxy features (e.g., Location as a proxy for socioeconomic status).

---

## 🤖 GenAI Usage in This Project

This simulation was designed to reflect modern AI-augmented analytics workflows:

| Task | GenAI Tool Used | Purpose |
|------|----------------|---------|
| EDA summarization | DeepSeek | Identify patterns, flag anomalies |
| Imputation strategy | ChatGPT | Best practice recommendations |
| Model scaffolding | DeepSeek | Generate initial pipeline structure |
| Fairness prompts | ChatGPT | Suggest bias detection frameworks |
| Code refinement | DeepSeek | Optimize and debug model code |

> *Note: All GenAI outputs were reviewed, validated, and refined manually. GenAI assisted the workflow — it did not replace analytical judgment.*

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/tata-genai-delinquency-prediction.git
cd tata-genai-delinquency-prediction

# 2. Install dependencies
pip install scikit-learn pandas numpy matplotlib seaborn

# 3. Run the pipeline
python tata_delinquency_model.py
```

Or open `tata_delinquency_model.ipynb` directly in **Google Colab** — no setup needed.

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **pandas / numpy** — Data manipulation
- **scikit-learn** — ML pipeline, model training, evaluation
- **matplotlib / seaborn** — Visualisation
- **GenAI tools** — DeepSeek, ChatGPT (workflow acceleration)

---

