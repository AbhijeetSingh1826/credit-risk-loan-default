# Credit Risk & Loan Default Prediction

A machine learning system that predicts loan default risk and translates predictions into business-relevant risk tiers and expected loss estimates, using real Lending Club loan data.

## Live Demo
🔗 [View Dashboard](https://credit-risk-loan-default-fhhzmf8dhc9a4cjxnvfrtm.streamlit.app/)

## Problem
Lenders need to estimate the likelihood a borrower will default before approving a loan, and translate that risk into an actual dollar cost. This project builds that full pipeline: from raw loan applications to a calibrated risk model to portfolio-level expected loss estimates.

## Approach
- **Data**: ~391K resolved Lending Club loans (2007-2018), using only features available at application time (no data leakage from post-approval fields)
- **Target definition**: Fully Paid vs. Charged Off/Default; excluded still-active loans (Current, Late, Grace Period) since their outcome isn't yet known
- **Preprocessing**: Handled missing values (median/Unknown-category imputation), ordinal encoding for ranked features (grade, sub_grade), one-hot encoding for unordered categories
- **Modeling**: Compared Logistic Regression (class-weighted), Random Forest, and XGBoost; evaluated with Precision, Recall, F1, and ROC-AUC due to class imbalance (~20% default rate)
- **Calibration**: Detected and corrected probability miscalibration caused by imbalance handling, using CalibratedClassifierCV, verified against actual default rates per risk tier
- **Risk Analytics**: Converted probabilities into 4 risk tiers and calculated Expected Loss (PD × EAD × LGD) at the portfolio level
- **Deployment**: Built and deployed a Streamlit dashboard for portfolio-level risk monitoring

## Results
| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression (weighted) | 0.346 | 0.635 | 0.448 | 0.729 |
| Random Forest | 0.337 | 0.662 | 0.447 | 0.731 |
| XGBoost (final model) | 0.342 | 0.668 | 0.452 | 0.735 |

**Portfolio-level findings** (test set, 78,234 loans):
- Total exposure: $1.15B
- Modeled expected loss: $163.3M (14.23% of exposure)
- Expected loss per borrower ranges from $671 (Low Risk) to $7,122 (Very High Risk)
- Top risk drivers: loan grade, term, sub-grade, interest rate — consistent with established credit risk fundamentals

## Tech Stack
Python, Pandas, Scikit-learn, XGBoost, Streamlit, Plotly

## Project Structure
\`\`\`
data/       - dataset and processed results (raw data gitignored)
notebooks/  - EDA, preprocessing, and model development
models/     - saved trained models and scaler
app/        - Streamlit dashboard
\`\`\`

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app/dashboard.py
```