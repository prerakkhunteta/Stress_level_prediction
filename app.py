import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Set page config
st.set_page_config(
    page_title="Anxiety & Stress Level Predictor",
    page_icon="🧠",
    layout="wide"
)

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv('anxiety_attack_dataset.csv')
    return df

@st.cache_resource
def train_model(df):
    features = [
        'Age', 'Sleep Hours', 'Physical Activity (hrs/week)',
        'Caffeine Intake (mg/day)', 'Alcohol Consumption (drinks/week)',
        'Heart Rate (bpm during attack)', 'Breathing Rate (breaths/min)',
        'Sweating Level (1-5)', 'Diet Quality (1-10)'
    ]
    X = df[features]
    y = df['Stress Level (1-10)']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, features

# Load data and train model once
try:
    df = load_data()
    model, features = train_model(df)
except Exception as e:
    st.error("Error loading the dataset. Please make sure the data file is present.")
    st.stop()

# Title and description
st.title("🧠 Anxiety & Stress Level Predictor")
st.markdown("""
    This app predicts stress levels using Random Forest algorithm based on various physical and lifestyle factors.
    Enter your information below to get a prediction.
""")

# Create three columns for input
st.subheader("📝 Enter Your Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Personal Details")
    age = st.number_input("Age", 18, 100, 30)
    sleep_hours = st.number_input("Sleep Hours", 0.0, 24.0, 7.0)
    physical_activity = st.number_input("Physical Activity (hrs/week)", 0.0, 40.0, 3.0)

with col2:
    st.markdown("### Lifestyle Factors")
    caffeine_intake = st.number_input("Caffeine Intake (mg/day)", 0, 1000, 200)
    alcohol_consumption = st.number_input("Alcohol Consumption (drinks/week)", 0, 50, 2)
    diet_quality = st.slider("Diet Quality", 1, 10, 5)

with col3:
    st.markdown("### Physical Symptoms")
    heart_rate = st.number_input("Heart Rate (bpm)", 40, 200, 80)
    breathing_rate = st.number_input("Breathing Rate (breaths/min)", 8, 40, 16)
    sweating_level = st.slider("Sweating Level", 1, 5, 3)

# Prediction button
if st.button("Predict Stress Level"):
    with st.spinner("Analyzing your data..."):
        # Prepare input data
        input_data = pd.DataFrame({
            'Age': [age],
            'Sleep Hours': [sleep_hours],
            'Physical Activity (hrs/week)': [physical_activity],
            'Caffeine Intake (mg/day)': [caffeine_intake],
            'Alcohol Consumption (drinks/week)': [alcohol_consumption],
            'Heart Rate (bpm during attack)': [heart_rate],
            'Breathing Rate (breaths/min)': [breathing_rate],
            'Sweating Level (1-5)': [sweating_level],
            'Diet Quality (1-10)': [diet_quality]
        })
        
        # Make prediction using cached model
        prediction = model.predict(input_data)[0]
        prediction = round(prediction * 2) / 2  # Round to nearest 0.5
        prediction = max(1, min(10, prediction))  # Ensure prediction is within bounds
        
        # Determine stress level and color
        if prediction <= 3:
            color = "#28a745"
        else:
            color = "#dc3545"
        
        # Display prediction
        st.markdown(f"""
            <div style='padding: 20px; border-radius: 10px; background-color: {color}; text-align: center; margin: 20px 0;'>
                <h2 style='color: white; margin: 0;'>Predicted Stress Level: {prediction:.1f}</h2>
            </div>
        """, unsafe_allow_html=True)
