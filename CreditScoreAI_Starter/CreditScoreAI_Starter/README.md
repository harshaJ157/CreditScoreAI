CreditScoreAI — Credit Score Category Prediction
CreditScoreAI is an educational machine-learning web application that predicts a credit-score category (Poor, Standard, or Good) from selected financial and credit-behaviour inputs. It is built with Python and Streamlit and includes Prediction, Credit Analysis, What-if Simulator, and Model Info views.
Important: This is not an official CIBIL score calculator or a financial decision system. The reported hold-out accuracy during development was approximately 56.4%, so predictions may be wrong and must not be used for real lending or financial decisions.

Features
- Predict a credit-score category from user-entered attributes.
- Review an educational profile summary and factor breakdown.
- Explore changes through a what-if simulator.
- View model information and visualizations.
Technology
- Python
- Streamlit
- pandas / NumPy
- scikit-learn
- joblib
- Plotly
Check the actual imports and requirements.txt to confirm package versions.
Input features
The project uses these candidate features:
- Outstanding_Debt
- Credit_Utilization_Ratio
- Credit_History_Age
- Payment_of_Min_Amount
- Total_EMI_per_month
- Amount_invested_monthly
- Payment_Behaviour
- Monthly_Balance
The target column is Credit_Score, with labels including Poor, Standard, and Good. See train_model.py for the exact feature processing.
Representative project structure
CreditScoreAI/
├── app.py
├── train_model.py
├── data/
│   ├── train.csv
│   └── test.csv
├── models/
│   ├── credit_score_model.joblib
│   └── metrics.json
├── utils/
│   ├── analysis.py
│   └── recommendations.py
├── requirements.txt
├── .gitignore
└── README.md
Adjust this to match your actual repository.
Setup on Windows
Open the folder containing app.py in VS Code, then run:
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
If requirements.txt does not exist yet, install the dependencies used by your project and create it:
python -m pip install streamlit pandas numpy scikit-learn joblib plotly
python -m pip freeze > requirements.txt
Review requirements.txt and remove packages that are not needed.

If the trained model is missing and the dataset is available in the paths expected by the script, train it:
python train_model.py
Run the application:
python -m streamlit run app.py
Streamlit will print a local address, commonly http://localhost:8501.
Workflow
1. Load the labelled dataset.
2. Prepare the selected input features and target labels.
3. Train and compare the configured machine-learning models.
4. Evaluate the selected model on a hold-out set.
5. Save the estimator and metrics.
6. Load the saved model in Streamlit.
7. Accept user inputs and display a predicted category with visual summaries.
Reported development results
- Training rows used: 100,000
- Selected model: Random Forest
- Hold-out accuracy: 0.564 (56.4%)
These are the reported results from the development run and can change with different data, preprocessing, or training settings. Add a confusion matrix, per-class precision/recall, and macro F1-score for a more complete evaluation.
Before publishing to GitHub
- Exclude venv/, .env, credentials, personal information, caches, and temporary files.
- Confirm that the dataset licence allows public redistribution. If not, exclude the CSV files and explain how to obtain them.
- Decide intentionally whether to include the .joblib model; it may be large and should only be published when appropriate.
- Do not exclude the model or data without documenting how a user can obtain or generate them.
Suggested .gitignore:
__pycache__/
*.py[cod]
.pytest_cache/
venv/
.venv/
env/
.env
*.key
.vscode/
.DS_Store
Thumbs.db
*.log
Limitations and responsible use
- The reported 56.4% hold-out accuracy is modest and not suitable for dependable real-world credit decisions.
- The model does not calculate an official CIBIL score.
- Any category-based visual indicator in the interface is illustrative, not a true numeric credit score or default probability.
- Avoid entering real personal or financial data into a public demo.
- Review data licensing, privacy, and model fairness before sharing or deploying.
Future improvements
- Add cross-validation and class-specific evaluation.
- Show a confusion matrix and classification report.
- Improve validation, preprocessing, and tests.
- Investigate model quality, class imbalance, fairness, and privacy.
License
Choose a licence only after deciding how the source code, dataset, and model may be reused. Dataset terms may differ from code terms.