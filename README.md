# LLM-Driven Credit Risk Assessment for Loan Approvals

This project explores the use of **machine learning and LLM-driven signals** for credit risk assessment.  
The workflow combines **structured loan features (LendingClub dataset)** with **text-derived borrower complaint signals (CFPB dataset)** to improve loan approval risk prediction.  

---

## Repository Structure

```text
src/
│── data_ingestion.py # Loads LendingClub & CFPB datasets into parquet format
│── add_cfpb_cohorts.py # Aggregates CFPB complaints by ZIP3 × Year, merges with LendingClub
│── visualize_metrics.py # Plots evaluation metrics (ROC, PR, Calibration, Confusion matrix)
│
├── eda/
│ ├── eda_lendingclub.ipynb # Exploratory analysis of LendingClub structured data
│ └── eda_cfpb_text.ipynb # Exploratory analysis of CFPB text data
│
├── features/
│ ├── features_numeric.py # Feature engineering on LendingClub (DTI, grade, cohorts, etc.)
│ └── features_text.py # Text feature extraction (sentiment, complaint counts)
│
├── models/
│ ├── train_numeric.py # Train baseline and engineered models using XGBoost
│ ├── tune_numeric.py # Hyperparameter tuning with Optuna
│ └── shap_analysis.py # Model interpretability via SHAP values
│
└── archive/
└── train_text.py # Unused (ID mismatch prevented merge of numeric & text rows)

```
---

## Workflow Overview

1. **Data Ingestion**  
   Structured LendingClub loans and CFPB complaint data are loaded and preprocessed into parquet format.  

2. **Feature Engineering**  
   - **Numeric features**: Loan grade, DTI, interest rate, repayment history.  
   - **Text features**: Sentiment & complaint counts aggregated at ZIP3 × Year cohorts.  

3. **Model Training**  
   XGBoost models are trained on:  
   - **Baseline numeric features only**  
   - **Engineered + CFPB text features**  

4. **Hyperparameter Tuning**  
   Optuna was used to optimize learning rate, max depth, child weight, etc.  

5. **Evaluation Metrics**  
   Models were assessed via:  
   - ROC Curve & AUC  
   - Precision-Recall Curve  
   - Confusion Matrix  
   - Calibration Curve  
   - SHAP feature importance  

---

## Key Findings

- Adding CFPB-derived sentiment and complaint counts **improved ROC-AUC and PR-AUC by ~1–2%** over the baseline model.  
- SHAP analysis confirmed that **temporal + complaint features** rank alongside traditional credit factors.  
- Calibration improved with Platt scaling, making probabilities more reliable for loan approval thresholds.  

---

## Tools and Libraries

- Python 3.11  
- **Pandas, NumPy** – data handling  
- **scikit-learn** – metrics, calibration, preprocessing  
- **XGBoost** – core classifier  
- **Optuna** – hyperparameter optimization  
- **SHAP** – interpretability  
- **Matplotlib/Seaborn** – plotting  
- **MLflow** – experiment logging and tracking  

---

## Project Proposal & Report

- `Project Proposal - LLM-Driven Credit Risk Assessment for Loan Approvals.pdf` – initial research plan  
- `MRP_Report.pdf` – final project report including methodology, experiments, and findings  

---
