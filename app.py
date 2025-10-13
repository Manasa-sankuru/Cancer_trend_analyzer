
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ------------------------
# Load model & preprocessing
# ------------------------
model = joblib.load(r"cancer_model.pkl")
scaler = joblib.load(r"scaler.pkl")
encoder = joblib.load(r"target_encoder.pkl")

# Load dataset
df = pd.read_csv(r"death .csv")
counties = df['County'].unique()

# ------------------------
# Page Config
# ------------------------
st.set_page_config(
    page_title="Cancer Trend Analyzer",
    page_icon="🎗️",
    layout="wide"
)

# ------------------------
# Custom Styling
# ------------------------
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(180deg, #e6f2f5 0%, #f8fdff 100%);
        font-family: 'Poppins', sans-serif;
        color:#001219;
    }
    .header {
        color: #002f4b;
        font-size: 34px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }
    .subheader {
        color: #006d77;
        font-size: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    .prediction-box {
        padding: 25px;
        border-radius: 15px;
        color: white;
        font-size: 22px;
        font-weight: 600;
        text-align: center;
        margin-top: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #006d77;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 0.6em 1.5em;
    }
    .stButton>button:hover {
        background-color: #004b50;
        color: #f0f0f0;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------
# Header
# ------------------------
st.markdown("<div class='header'>Cancer Trend Analyzer</div>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Predict county-level cancer mortality trends using Machine Learning</p>", unsafe_allow_html=True)
st.markdown("---")

# ------------------------
# Sidebar
# ------------------------
with st.sidebar:
    st.header("About")
    st.write("""
    This dashboard predicts **county-level cancer mortality trends**
    using a trained **Random Forest Regressor** model.
    Enter your data below to visualize trend predictions.
    """)
    st.markdown("---")
    st.write("**Dataset:** death.csv")
    st.write("**Model:** Random Forest Regressor")
    st.write("**Developed by:** Manasa Sankuru")
    st.markdown("---")
    st.header("Contact")
    st.write("Email: your_email@example.com")

# ------------------------
# Input Section
# ------------------------
st.header("Input Data")

col1, col2 = st.columns(2)

with col1:
    county = st.selectbox(
        "Country",
        counties,
        help="Select the county for which you want to predict cancer mortality trends."
    )
    index_val = st.number_input(
        "Index",
        value=0,
        help="Unique index number for this record (usually from your dataset)."
    )
    FIPS = st.number_input(
        "FIPS Code",
        value=0,
        help="Federal Information Processing Standard code for the county."
    )
    met_objective = st.selectbox(
        "Met Objective (45.5)",
        ["Yes", "No"],
        help="Select 'Yes' if the county met the target death rate of 45.5, otherwise 'No'."
    )
    age_adjusted = st.number_input(
        "Age-Adjusted Death Rate",
        value=0.0,
        help="Deaths per 100,000 people, adjusted for age distribution."
    )

with col2:
    lower_ci_death = st.number_input(
        "Lower 95% CI for Death Rate",
        value=0.0,
        help="Lower bound of the 95% confidence interval for the death rate."
    )
    upper_ci_death = st.number_input(
        "Upper 95% CI for Death Rate",
        value=0.0,
        help="Upper bound of the 95% confidence interval for the death rate."
    )
    avg_deaths = st.number_input(
        "Average Deaths per Year",
        value=0.0,
        help="Average number of deaths per year in this county."
    )
    recent_trend_rate = st.number_input(
        "Recent 5-Year Trend",
        value=0.0,
        help="Change in death rates over the past 5 years. Positive = rising, Negative = falling."
    )
    lower_ci_trend = st.number_input(
        "Lower 95% CI for Trend",
        value=0.0,
        help="Lower bound of the 95% confidence interval for the trend estimate."
    )
    upper_ci_trend = st.number_input(
        "Upper 95% CI for Trend",
        value=0.0,
        help="Upper bound of the 95% confidence interval for the trend estimate."
    )

# ------------------------
# Predict Button
# ------------------------
predict_button = st.button("Predict Trend")

if predict_button:
    met_objective_val = 1 if met_objective == "Yes" else 0
    county_encoded = county.__hash__() % 100000

    input_df = pd.DataFrame([[index_val, county_encoded, FIPS, met_objective_val, age_adjusted,
                              lower_ci_death, upper_ci_death, avg_deaths, recent_trend_rate,
                              lower_ci_trend, upper_ci_trend]],
                            columns=[
                                'index', 'County', 'FIPS', 'Met Objective of 45.5? (1)',
                                'Age-Adjusted Death Rate', 'Lower 95% Confidence Interval for Death Rate',
                                'Upper 95% Confidence Interval for Death Rate', 'Average Deaths per Year',
                                'Recent 5-Year Trend (2) in Death Rates', 'Lower 95% Confidence Interval for Trend',
                                'Upper 95% Confidence Interval for Trend'
                            ])

    input_scaled = input_df.copy()
    input_scaled[input_df.columns] = scaler.transform(input_df[input_df.columns])
    prediction = model.predict(input_scaled)[0]

    # Color & Status
    if prediction < -0.5:
        status = "Falling"
        box_color = "#005f73"
    elif prediction > 0.5:
        status = "Rising"
        box_color = "#0a9396"
    else:
        status = "Stable"
        box_color = "#94d2bd"

    st.subheader("Prediction Result")
    st.markdown(
        f"<div class='prediction-box' style='background-color:{box_color}'>{status} ({prediction:.2f})</div>",
        unsafe_allow_html=True
    )

    # Comparison chart
    df['Recent 5-Year Trend (2) in Death Rates'] = pd.to_numeric(
    df['Recent 5-Year Trend (2) in Death Rates'], errors='coerce'
)

    national_avg_trend = df['Recent 5-Year Trend (2) in Death Rates'].mean(skipna=True)
    chart_data = pd.DataFrame({
        'Metric': ['County Trend', 'National Avg'],
        'Trend': [prediction, national_avg_trend]
    })
    fig = px.bar(chart_data, x='Metric', y='Trend', color='Metric',
                 color_discrete_sequence=["#0a9396", "#005f73"],
                 title="County vs National Average")
    st.plotly_chart(fig, use_container_width=True)

    # Display Inputs
    st.markdown("### Entered Inputs")
    st.dataframe(input_df.style.set_properties(**{
        'background-color': '#e9f5f6',
        'color': '#003049'
    }))

