from pathlib import Path
import json
import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import plotly.express as px
from utils.analysis import profile_to_frame
from utils.recommendations import generate_recommendations

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "credit_score_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

st.set_page_config(page_title="CreditScore AI", page_icon="📊", layout="wide")

st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top left, #123b40 0%, transparent 38%),
                linear-gradient(135deg, #08111f 0%, #0d1928 55%, #102a2c 100%);
    color: #edf5f7;
}
.block-container { padding-top: 2rem; max-width: 1450px; }
h1, h2, h3 { color: #f8fafc !important; }
.stTabs [data-baseweb="tab-list"] { gap: 10px; border-bottom: 1px solid #294253; }
.stTabs [aria-selected="true"] { color: #5eead4 !important; }
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #14283a, #102b32);
    border: 1px solid #2d4b5b;
    border-radius: 14px;
    padding: 18px;
}
div[data-testid="stMetricValue"] { color: #5eead4; }
.stButton > button, .stFormSubmitButton > button {
    background: linear-gradient(90deg, #0d9488, #0f766e);
    color: white; border: 1px solid #2dd4bf; border-radius: 10px;
    padding: 0.6rem 1.2rem; font-weight: 600;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    background: #14b8a6; color: white;
}
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
    background: #132437; border: 1px solid #304a5c; border-radius: 10px;
}
@media (max-width: 768px) {
    .block-container { padding: 1rem 0.8rem 2rem; }
}
</style>
""", unsafe_allow_html=True)

st.title("📊 CreditScore AI")
st.caption("Machine-learning credit category analysis • Educational prototype")

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run `python train_model.py` from this project folder.")
    st.stop()

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
features = artifact["features"]

metrics = {}
if METRICS_PATH.exists():
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

st.subheader("Dashboard Overview")
overview1, overview2, overview3 = st.columns(3)
overview1.metric("Selected Model", artifact.get("model_name", "Unknown"))
overview2.metric("Training Records", f"{artifact.get('dataset_rows', 0):,}")
accuracy = metrics.get("best_accuracy")
overview3.metric("Hold-out Accuracy", f"{accuracy * 100:.1f}%" if accuracy is not None else "N/A")
st.divider()

tab_predict, tab_analyze, tab_simulator, tab_about = st.tabs([
    "🔮 Prediction", "🧭 Credit Analysis", "🔄 What-if Simulator", "ℹ️ Model Info"
])

NUMERIC_RANGES = {
    "Outstanding_Debt": (0.0, 500000.0, 25000.0),
    "Credit_Utilization_Ratio": (0.0, 100.0, 30.0),
    "Total_EMI_per_month": (0.0, 200000.0, 5000.0),
    "Amount_invested_monthly": (0.0, 200000.0, 5000.0),
    "Monthly_Balance": (0.0, 200000.0, 10000.0),
}
CATEGORY_OPTIONS = {
    "Credit_History_Age": [
        "0 Years and 0 Months", "1 Years and 0 Months", "2 Years and 0 Months",
        "5 Years and 0 Months", "10 Years and 0 Months"
    ],
    "Payment_of_Min_Amount": ["Yes", "No", "NM"],
    "Payment_Behaviour": [
        "Low_spent_Small_value_payments", "Low_spent_Medium_value_payments",
        "Low_spent_Large_value_payments", "High_spent_Small_value_payments",
        "High_spent_Medium_value_payments", "High_spent_Large_value_payments"
    ],
}

def input_profile(prefix):
    values = {}
    columns = st.columns(2)
    for i, feature in enumerate(features):
        with columns[i % 2]:
            if feature in NUMERIC_RANGES:
                low, high, default = NUMERIC_RANGES[feature]
                values[feature] = st.number_input(
                    feature.replace("_", " "),
                    min_value=float(low),
                    max_value=float(high),
                    value=float(default),
                    step=1.0,
                    key=f"{prefix}_{feature}",
                )
            else:
                options = CATEGORY_OPTIONS.get(feature, ["Unknown"])
                values[feature] = st.selectbox(
                    feature.replace("_", " "), options, key=f"{prefix}_{feature}"
                )
    return values


with tab_predict:
    # ---------- FINTECH PREDICTION DASHBOARD ----------
    st.markdown("""
    <style>
    .prediction-hero {
        background: linear-gradient(125deg, #142b3d 0%, #123d43 100%);
        border: 1px solid #28616a;
        padding: 24px;
        border-radius: 18px;
        margin: 8px 0 22px 0;
    }
    .prediction-hero h2 {
        color: #e9fffa;
        margin: 0 0 8px 0;
    }
    .prediction-hero p {
        color: #b8d4d8;
        margin: 0;
    }
    .result-card {
        background: linear-gradient(135deg, #122638, #153a42);
        border: 1px solid #32616b;
        border-radius: 18px;
        padding: 24px;
        margin: 14px 0;
    }
    .result-label {
        color: #a9c7d0;
        font-size: 14px;
        margin-bottom: 8px;
    }
    .result-value {
        font-size: 36px;
        font-weight: 750;
        line-height: 1.2;
        margin-bottom: 10px;
    }
    .result-note {
        color: #c4d9df;
        font-size: 14px;
    }
    .section-note {
        color: #9fb7c5;
        font-size: 13px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="prediction-hero">
        <h2>Credit Profile Prediction</h2>
        <p>Analyze the information you provide and explore your model's
        predicted credit category, probability distribution, and next steps.</p>
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        "Educational prototype only. This model predicts dataset categories; "
        "it does not calculate an official CIBIL score or determine actual loan eligibility."
    )

    # ---------- INPUT FORM ----------
    st.markdown("### 1. Enter your profile")
    st.markdown(
        '<p class="section-note">Use approximate values for testing. '
        'Do not enter account numbers, passwords, or OTPs.</p>',
        unsafe_allow_html=True
    )

    numeric_features = [f for f in features if f in NUMERIC_RANGES]
    categorical_features = [f for f in features if f not in NUMERIC_RANGES]

    with st.form("prediction_form"):
        left, right = st.columns(2, gap="large")

        with left:
            st.markdown("#### Financial details")
            prediction_profile = {}

            for feature in numeric_features:
                low, high, default = NUMERIC_RANGES[feature]
                prediction_profile[feature] = st.number_input(
                    feature.replace("_", " "),
                    min_value=float(low),
                    max_value=float(high),
                    value=float(default),
                    step=1.0,
                    key=f"prediction_{feature}",
                    help={
                        "Outstanding_Debt": "Approximate outstanding debt amount.",
                        "Credit_Utilization_Ratio": "Percentage of available revolving credit being used.",
                        "Total_EMI_per_month": "Total monthly EMI payments.",
                        "Amount_invested_monthly": "Approximate monthly investment amount.",
                        "Monthly_Balance": "Approximate monthly balance."
                    }.get(feature, "Enter an approximate value.")
                )

        with right:
            st.markdown("#### Credit behaviour")
            for feature in categorical_features:
                options = CATEGORY_OPTIONS.get(feature, ["Unknown"])
                prediction_profile[feature] = st.selectbox(
                    feature.replace("_", " "),
                    options,
                    key=f"prediction_{feature}"
                )

        submitted = st.form_submit_button(
            "🔍 Analyze Credit Profile",
            type="primary",
            use_container_width=True
        )

    # ---------- PREDICTION RESULTS ----------
    if submitted:
        row = profile_to_frame(prediction_profile, features)
        prediction = str(model.predict(row)[0])
        normalized_prediction = prediction.strip().lower()

        # The meter is an illustrative category-based indicator.
        # It is not a validated probability of default or real credit risk.
        category_meter = {
            "good": 20,
            "standard": 50,
            "poor": 80
        }
        meter_value = category_meter.get(normalized_prediction)

        category_colors = {
            "good": "#42e6b5",
            "standard": "#f6c667",
            "poor": "#ff7b86"
        }
        category_color = category_colors.get(
            normalized_prediction, "#67d9e8"
        )

        st.divider()
        st.markdown("### 2. Prediction results")

        result_col, details_col = st.columns([1.1, 1], gap="large")

        with result_col:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Predicted credit category</div>
                <div class="result-value" style="color:{category_color}">
                    {prediction}
                </div>
                <div class="result-note">
                    Based on patterns learned from the training dataset.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with details_col:
            st.markdown("#### Model summary")
            st.metric(
                "Selected model",
                artifact.get("model_name", "Trained classifier")
            )
            st.metric(
                "Training records",
                f"{artifact.get('dataset_rows', 0):,}"
            )

            if metrics and "best_accuracy" in metrics:
                st.metric(
                    "Hold-out accuracy",
                    f"{metrics['best_accuracy'] * 100:.1f}%"
                )
                st.caption(
                    "Dataset evaluation metric; not a measure of this "
                    "individual prediction's correctness."
                )

        # ---------- EDUCATIONAL RISK METER ----------
        st.markdown("### 3. Category-based indicator")

        if meter_value is not None:
            st.markdown(
                f"**Illustrative indicator: {meter_value}/100**"
            )
            st.progress(meter_value / 100)

            st.markdown(
                f"""
                <p class="section-note">
                This demonstration maps the predicted category
                <b>{prediction}</b> to a fixed display value. It does not
                measure a person's actual credit risk, default probability,
                or official CIBIL score.
                </p>
                """,
                unsafe_allow_html=True
            )
        else:
            st.info(
                "A category-based indicator is not configured for this "
                "model's predicted label. The predicted category is still shown above."
            )

        # ---------- MODEL PROBABILITIES ----------
        st.markdown("### 4. Model probability distribution")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(row)[0]
            labels = [str(label) for label in model.classes_]

            prob_df = pd.DataFrame({
                "Category": labels,
                "Probability (%)": [
                    round(float(p) * 100, 2) for p in probabilities
                ]
            }).sort_values("Probability (%)", ascending=True)

            fig = px.bar(
                prob_df,
                x="Probability (%)",
                y="Category",
                orientation="h",
                text="Probability (%)",
                color="Probability (%)",
                color_continuous_scale=["#23465a", "#1aa899", "#65edc5"],
                range_color=[0, 100],
                title="How the model distributes its predictions"
            )

            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside",
                cliponaxis=False
            )
            fig.update_layout(
                height=300,
                showlegend=False,
                coloraxis_showscale=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e6f1f5",
                xaxis=dict(
                    range=[0, min(110, max(20, float(prob_df["Probability (%)"].max()) + 12))],
                    title="Model probability (%)",
                    gridcolor="rgba(150,180,200,0.15)"
                ),
                yaxis=dict(title=""),
                margin=dict(l=10, r=30, t=55, b=20)
            )

            st.plotly_chart(fig, use_container_width=True)

            top_row = prob_df.iloc[-1]
            st.caption(
                f"Highest model probability: {top_row['Category']} "
                f"({top_row['Probability (%)']:.1f}%). "
                "These are model outputs and may not be calibrated."
            )
        else:
            st.info(
                "This model does not provide predict_proba outputs, "
                "so a probability chart is unavailable."
            )

        # ---------- PERSONALIZED NEXT STEPS ----------
        st.markdown("### 5. Suggested next steps")

        recommendations = generate_recommendations(prediction_profile)

        if recommendations:
            for i, tip in enumerate(recommendations, start=1):
                st.markdown(f"**{i}.** {tip}")
        else:
            st.write(
                "Continue monitoring payment due dates, outstanding balances, "
                "and your official credit report."
            )

        st.info(
            "A prediction of Standard does not mean your profile is necessarily "
            "average in every area. Review the inputs, recommendations, and "
            "your official credit report for a fuller picture."
        )

    else:
        # Helpful initial state before the user submits the form.
        st.markdown("### What you'll see here")
        preview1, preview2, preview3 = st.columns(3)

        preview1.markdown("**Prediction**")
        preview1.caption("The model's predicted category.")

        preview2.markdown("**Probability chart**")
        preview2.caption("The model's distribution across available categories.")

        preview3.markdown("**Recommendations**")
        preview3.caption("General suggestions based on the entered profile.")

with tab_analyze:
    st.subheader("Credit Health Checklist")
    st.write("General education only; this is not a bureau score formula.")
    utilization = st.slider("Credit utilization (%)", 0, 100, 30)
    payment = st.selectbox("Do you usually pay bills by the due date?", ["Yes", "Sometimes", "No"])
    history = st.selectbox("Credit history", ["No established history", "Less than 2 years", "2–5 years", "More than 5 years"])
    debt = st.selectbox("Outstanding debt relative to income", ["Low", "Moderate", "High", "Not sure"])

    health = 0
    health += 35 if utilization <= 30 else 15
    health += 35 if payment == "Yes" else 20 if payment == "Sometimes" else 5
    health += 20 if debt == "Low" else 12 if debt == "Moderate" else 5
    health += 10 if history == "More than 5 years" else 7 if history == "2–5 years" else 4

    st.metric("Educational Checklist Indicator", f"{health}/100")
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=health,
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#14b8a6"},
            "steps": [
                {"range": [0, 40], "color": "#4b252e"},
                {"range": [40, 70], "color": "#5b4927"},
                {"range": [70, 100], "color": "#17483e"},
            ],
        },
    ))
    gauge.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", font_color="#edf5f7")
    st.plotly_chart(gauge, use_container_width=True)

    st.markdown("**Suggestions**")
    if utilization > 30:
        st.write("• Consider reducing revolving credit balances if practical; 30% is a common guideline, not a universal cutoff.")
    else:
        st.write("• Lower utilization can be helpful, but it does not guarantee a higher credit score.")
    if payment != "Yes":
        st.write("• Prioritize paying bills by their due dates and contact lenders early if you expect difficulty.")
    if history == "No established history":
        st.write("• A limited credit history is not automatically a poor credit history.")
    if debt == "High":
        st.write("• Review your repayment commitments and budget before taking on additional borrowing.")
    st.write("• Review your official credit report periodically and dispute inaccurate entries.")

with tab_simulator:
    st.subheader("What-if Credit Category Simulator")
    st.write("Compare two hypothetical profiles. Outputs are model predictions, not guaranteed real-world changes.")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Scenario A")
        profile_a = input_profile("sim_a")
    with right:
        st.markdown("#### Scenario B")
        profile_b = input_profile("sim_b")

    
    if st.button("Compare Scenarios", type="primary"):
        row_a = profile_to_frame(profile_a, features)
        row_b = profile_to_frame(profile_b, features)

        prediction_a = model.predict(row_a)[0]
        prediction_b = model.predict(row_b)[0]

        col_a, col_b = st.columns(2)

        with col_a:
            st.metric("Scenario A Category", str(prediction_a))
            if hasattr(model, "predict_proba"):
                probs_a = model.predict_proba(row_a)[0]
                st.caption("Model probability distribution")
                st.bar_chart(
                    pd.DataFrame({
                        "Probability (%)": probs_a * 100
                    }, index=model.classes_)
                )

        with col_b:
            st.metric("Scenario B Category", str(prediction_b))
            if hasattr(model, "predict_proba"):
                probs_b = model.predict_proba(row_b)[0]
                st.caption("Model probability distribution")
                st.bar_chart(
                    pd.DataFrame({
                        "Probability (%)": probs_b * 100
                    }, index=model.classes_)
                )

        if hasattr(model, "predict_proba"):
            probs_a = model.predict_proba(row_a)[0]
            probs_b = model.predict_proba(row_b)[0]

            comparison = pd.DataFrame({
                "Category": model.classes_,
                "Scenario A (%)": (probs_a * 100).round(1),
                "Scenario B (%)": (probs_b * 100).round(1),
            })
            comparison["Difference (percentage points)"] = (
                comparison["Scenario B (%)"] - comparison["Scenario A (%)"]
            ).round(1)

            st.subheader("Probability comparison")
            st.dataframe(comparison, hide_index=True, use_container_width=True)

        if prediction_a == prediction_b:
            st.info(
                "Both scenarios received the same predicted category. "
                "Their probability distributions may still differ."
            )
        else:
            st.success("The model predicted different categories for the two scenarios.")

        st.caption(
            "These are model outputs, not official CIBIL scores. "
            "Changing an input does not guarantee a different category."
        )


with tab_about:
    st.subheader("Model Information")
    info1, info2 = st.columns(2)
    info1.write(f"**Selected model:** {artifact.get('model_name', 'Unknown')}")
    info2.write(f"**Rows used:** {artifact.get('dataset_rows', 'Unknown')}")
    if metrics:
        st.metric("Hold-out Accuracy", f"{metrics.get('best_accuracy', 0) * 100:.1f}%")
        st.caption("Measured on a held-out dataset split; it does not establish real-world or official CIBIL accuracy.")
        st.json({key: value for key, value in metrics.items() if key != "results"})
    st.markdown("""
**Limitations**
- This is an educational prototype, not a financial or lending decision system.
- Results depend on the dataset, its quality, representativeness, and labels.
- It does not calculate an individual's official CIBIL score.
- Do not use this prototype to approve or deny credit.
""")
