import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Set page config
st.set_page_config(
    page_title="Anxiety Attack Severity Predictor",
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
    y = df['Severity of Anxiety Attack (1-10)']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Get feature importance
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Select top 5 most important features
    top_features = feature_importance['feature'].head(5).tolist()
    
    # Retrain model with only important features
    X = df[top_features]
    model.fit(X, y)
    return model, top_features

# Load data and train model once
try:
    df = load_data()
    model, features = train_model(df)
    
    # Display feature importance in sidebar
    st.sidebar.header("Most Important Factors")
    st.sidebar.write("These are the top 5 factors that influence anxiety attack severity:")
    for feature in features:
        st.sidebar.markdown(f"- {feature}")
except Exception as e:
    st.error("Error loading the dataset. Please make sure the data file is present.")
    st.stop()

# Title and description
st.title("🧠 Anxiety Attack Severity Predictor")
st.markdown("""
    This app predicts anxiety attack severity using Random Forest algorithm based on various physical and lifestyle factors.
    Enter your information below to get a prediction.
""")

# Create input fields for important features only
st.header('Please Enter Your Information')

col1, col2 = st.columns(2)

# Create dictionary mapping features to their input functions
input_fields = {
    'Age': lambda: st.number_input('Age', min_value=0, max_value=100, value=30),
    'Sleep Hours': lambda: st.slider('Sleep Hours', 0.0, 12.0, 7.0, 0.1),
    'Physical Activity (hrs/week)': lambda: st.number_input('Physical Activity (hrs/week)', min_value=0.0, max_value=40.0, value=3.0),
    'Caffeine Intake (mg/day)': lambda: st.number_input('Caffeine Intake (mg/day)', min_value=0, max_value=1000, value=100),
    'Alcohol Consumption (drinks/week)': lambda: st.number_input('Alcohol Consumption (drinks/week)', min_value=0, max_value=50, value=0),
    'Heart Rate (bpm during attack)': lambda: st.number_input('Heart Rate (bpm during attack)', min_value=40, max_value=200, value=80),
    'Breathing Rate (breaths/min)': lambda: st.number_input('Breathing Rate (breaths/min)', min_value=10, max_value=50, value=20),
    'Sweating Level (1-5)': lambda: st.slider('Sweating Level', 1, 5, 3),
    'Diet Quality (1-10)': lambda: st.slider('Diet Quality', 1, 10, 5)
}

# Display only the important features
values = {}
for i, feature in enumerate(features):
    with col1 if i < len(features)//2 else col2:
        values[feature] = input_fields[feature]()

# Prediction button
if st.button("Predict Anxiety Attack Severity"):
    with st.spinner("Analyzing your data..."):
        # Prepare input data
        input_data = pd.DataFrame({feature: [values[feature]] for feature in features})
        
        # Make prediction using cached model
        prediction = model.predict(input_data)[0]
        prediction = round(prediction * 2) / 2  # Round to nearest 0.5
        prediction = max(1, min(10, prediction))  # Ensure prediction is within bounds
        
        # Determine severity level, color, and recommendations
        if prediction <= 3:
            color = "#28a745"  # Green for low severity
            severity_text = "Low Severity"
            recommendations = [
                "• Practice regular mindfulness or meditation to maintain your current mental well-being",
                "• Continue your existing healthy lifestyle habits",
                "• Monitor any changes in your anxiety levels",
                "• Maintain regular sleep schedule and exercise routine"
            ]
        elif prediction <= 6:
            color = "#ffc107"  # Yellow for medium severity
            severity_text = "Moderate Severity"
            recommendations = [
                "• Consider consulting a mental health professional",
                "• Practice deep breathing exercises during anxiety episodes",
                "• Reduce caffeine and alcohol intake",
                "• Establish a regular sleep schedule",
                "• Try stress-reduction techniques like yoga or meditation"
            ]
        else:
            color = "#dc3545"  # Red for high severity
            severity_text = "High Severity"
            recommendations = [
                "• Seek immediate professional mental health support",
                "• Consider discussing medication options with a healthcare provider",
                "• Develop an anxiety management plan with a professional",
                "• Practice emergency coping techniques",
                "• Ensure family/friends are aware to provide support",
                "• Consider joining support groups"
            ]
        
        # Display prediction and recommendations
        st.markdown(f"""
            <div style='padding: 20px; border-radius: 10px; background-color: {color}; text-align: center; margin: 20px 0;'>
                <h2 style='color: white; margin: 0;'>Predicted Anxiety Attack Severity: {prediction:.1f}</h2>
                <p style='color: white; margin: 10px 0;'>Level: {severity_text}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Display recommendations in a clean format
        st.markdown("### Recommendations")
        st.markdown("Based on the predicted severity level, here are some recommendations:")
        for rec in recommendations:
            st.markdown(rec)
            
        # Add disclaimer
        st.markdown("""
        ---
        **Disclaimer:** These recommendations are general guidelines and not a substitute for professional medical advice. 
        Always consult with qualified healthcare professionals for personalized medical advice.
        """)

# Add some information about the app
st.markdown('''
### About this app
This app helps predict the severity of anxiety attacks based on various personal and health-related factors.
Please consult with a healthcare professional for medical advice.
''')
