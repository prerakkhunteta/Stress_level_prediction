import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Set page config
st.set_page_config(
    page_title="Anxiety & Stress Level Predictor",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 20px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🧠 Anxiety & Stress Level Predictor")
st.markdown("""
    This app helps predict stress levels based on various physical and lifestyle factors.
    Enter your information below to get a prediction.
""")

# Load and cache data
@st.cache_data
def load_data():
    df = pd.read_csv('anxiety_attack_dataset.csv')
    return df

# Load and prepare the model
@st.cache_resource
def prepare_model(df, feature_cols):
    X = df[feature_cols]
    y = df['Stress Level (1-10)']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train the model with more trees and better parameters
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

# Load the data silently
try:
    df = load_data()
except Exception as e:
    st.error("Error loading the dataset. Please make sure the data file is present.")
    st.stop()

# Define feature columns
feature_cols = ['Age', 'Sleep Hours', 'Physical Activity (hrs/week)', 
               'Caffeine Intake (mg/day)', 'Alcohol Consumption (drinks/week)',
               'Heart Rate (bpm during attack)', 'Breathing Rate (breaths/min)',
               'Sweating Level (1-5)', 'Diet Quality (1-10)']

# Prepare the model
model, scaler = prepare_model(df, feature_cols)

# Create three columns for input
st.subheader("📝 Enter Your Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Personal Details")
    age = st.number_input("Age", 18, 100, 30)
    sleep_hours = st.number_input("Sleep Hours", 0.0, 24.0, 7.0, help="Average hours of sleep per day")
    physical_activity = st.number_input("Physical Activity (hrs/week)", 0.0, 40.0, 3.0, 
                                      help="Hours spent on physical activity per week")

with col2:
    st.markdown("### Lifestyle Factors")
    caffeine_intake = st.number_input("Caffeine Intake (mg/day)", 0, 1000, 200,
                                     help="Daily caffeine consumption in milligrams")
    alcohol_consumption = st.number_input("Alcohol Consumption (drinks/week)", 0, 50, 2,
                                        help="Number of alcoholic drinks per week")
    diet_quality = st.slider("Diet Quality", 1, 10, 5,
                            help="Rate your diet quality from 1 (poor) to 10 (excellent)")

with col3:
    st.markdown("### Physical Symptoms")
    heart_rate = st.number_input("Heart Rate (bpm)", 40, 200, 80,
                                help="Heart rate during anxiety attack")
    breathing_rate = st.number_input("Breathing Rate (breaths/min)", 8, 40, 16,
                                   help="Breathing rate during anxiety attack")
    sweating_level = st.slider("Sweating Level", 1, 5, 3,
                              help="Rate your sweating level from 1 (minimal) to 5 (excessive)")

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
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Round to nearest 0.5
        prediction = round(prediction * 2) / 2
        
        # Ensure prediction is within bounds
        prediction = max(1, min(10, prediction))
        
        # Define stress level categories and recommendations
        if prediction <= 3:
            level_text, color = "Low", "#28a745"
            recommendations = [
                "Continue maintaining your healthy lifestyle",
                "Practice regular relaxation techniques",
                "Keep up with your current stress management strategies"
            ]
        elif prediction <= 6:
            level_text, color = "Moderate", "#ffc107"
            recommendations = [
                "Consider increasing physical activity",
                "Practice deep breathing exercises",
                "Maintain a regular sleep schedule",
                "Consider reducing caffeine intake"
            ]
        else:
            level_text, color = "High", "#dc3545"
            recommendations = [
                "Consider consulting a healthcare professional",
                "Implement stress reduction techniques immediately",
                "Review and adjust lifestyle factors",
                "Ensure adequate rest and sleep",
                "Consider reducing work/study load if possible"
            ]
        
        # Display prediction and recommendations
        st.markdown(f"""
            <div style='padding: 20px; border-radius: 10px; background-color: {color}; text-align: center; margin: 20px 0;'>
                <h2 style='color: white; margin: 0;'>Predicted Stress Level: {prediction:.1f} ({level_text})</h2>
            </div>
        """, unsafe_allow_html=True)

        # Display feature importance
        st.subheader("Most Influential Factors:")
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)

        top_features = feature_importance.head(3)
        for idx, row in top_features.iterrows():
            st.markdown(f"- **{row['Feature']}**: {row['Importance']:.2%} influence on prediction")
        
        # Display recommendations
        st.subheader("📋 Recommendations:")
        for rec in recommendations:
            st.markdown(f"- {rec}")

        # Display confidence note
        st.markdown("""
        ---
        **Note**: This prediction is based on the patterns found in our dataset. 
        For accurate medical advice, please consult with a healthcare professional.
        """)