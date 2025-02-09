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

# Load data
try:
    df = pd.read_csv('anxiety_attack_dataset.csv')
except Exception as e:
    st.error("Error loading the dataset. Please make sure the data file is present.")
    st.stop()

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
        # Prepare the features
        feature_cols = [
            'Age', 'Sleep Hours', 'Physical Activity (hrs/week)',
            'Caffeine Intake (mg/day)', 'Alcohol Consumption (drinks/week)',
            'Heart Rate (bpm during attack)', 'Breathing Rate (breaths/min)',
            'Sweating Level (1-5)', 'Diet Quality (1-10)'
        ]
        
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
        
        # Prepare training data
        X = df[feature_cols]
        y = df['Stress Level (1-10)']
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        input_scaled = scaler.transform(input_data)
        
        # Train the model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        model.fit(X_scaled, y)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction = round(prediction * 2) / 2  # Round to nearest 0.5
        prediction = max(1, min(10, prediction))  # Ensure prediction is within bounds
        
        # Calculate feature importance scores
        importances = model.feature_importances_
        feature_imp = pd.DataFrame({
            'Feature': feature_cols,
            'Value': [input_data[col].iloc[0] for col in feature_cols],
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Generate dynamic recommendations based on feature importance and values
        recommendations = []
        
        # Add recommendations based on most important features
        for _, row in feature_imp.head(3).iterrows():
            feature = row['Feature']
            value = row['Value']
            
            if feature == 'Sleep Hours':
                if value < 6:
                    recommendations.append(f"Increase your sleep duration (currently {value:.1f} hours) to at least 7-8 hours")
                elif value > 9:
                    recommendations.append(f"Consider reducing sleep duration (currently {value:.1f} hours) to 7-8 hours")
            
            elif feature == 'Heart Rate (bpm during attack)':
                if value > 100:
                    recommendations.append(f"Your heart rate ({value:.0f} bpm) is elevated. Practice deep breathing exercises")
                elif value < 60:
                    recommendations.append(f"Your heart rate ({value:.0f} bpm) is low. Consider gentle exercise")
            
            elif feature == 'Caffeine Intake (mg/day)':
                if value > 400:
                    recommendations.append(f"Reduce caffeine intake (currently {value:.0f} mg/day) to below 400mg")
                elif value > 200:
                    recommendations.append(f"Consider moderating caffeine consumption (currently {value:.0f} mg/day)")
            
            elif feature == 'Physical Activity (hrs/week)':
                if value < 2:
                    recommendations.append(f"Increase physical activity (currently {value:.1f} hrs/week) to at least 2-3 hours")
                elif value > 15:
                    recommendations.append(f"Consider balancing your high physical activity ({value:.1f} hrs/week)")
        
        # Add stress level specific recommendations
        if prediction <= 3:
            level_text, color = "Low", "#28a745"
            recommendations.extend([
                f"Your stress level ({prediction:.1f}) indicates good management",
                "Continue your current wellness practices",
                "Consider preventive stress management techniques"
            ])
        elif prediction <= 6:
            level_text, color = "Moderate", "#ffc107"
            recommendations.extend([
                f"Your stress level ({prediction:.1f}) indicates room for improvement",
                "Consider daily mindfulness or meditation practice",
                "Evaluate and adjust your daily routine"
            ])
        else:
            level_text, color = "High", "#dc3545"
            recommendations.extend([
                f"Your stress level ({prediction:.1f}) requires attention",
                "Consider professional stress management support",
                "Implement immediate stress reduction strategies"
            ])
        
        # Display prediction
        st.markdown(f"""
            <div style='padding: 20px; border-radius: 10px; background-color: {color}; text-align: center; margin: 20px 0;'>
                <h2 style='color: white; margin: 0;'>Predicted Stress Level: {prediction:.1f} ({level_text})</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Display feature importance with actual values
        st.subheader("Most Influential Factors:")
        for idx, row in feature_imp.head(3).iterrows():
            st.markdown(f"- **{row['Feature']}** (Current value: {row['Value']:.1f})")
        
        # Display recommendations
        st.subheader("📋 Personalized Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")
        
        # Display analysis summary
        st.markdown(f"""
        ---
        **Analysis Summary**: 
        Your stress level prediction of {prediction:.1f} is primarily influenced by your {feature_imp.iloc[0]['Feature'].lower()} 
        (current value: {feature_imp.iloc[0]['Value']:.1f}). 
        
        **Key Factors Contributing to Your Stress Level:**
        - {feature_imp.iloc[0]['Feature']}: {'High' if feature_imp.iloc[0]['Value'] > np.mean(df[feature_imp.iloc[0]['Feature']]) else 'Low'} relative to average
        - {feature_imp.iloc[1]['Feature']}: {'High' if feature_imp.iloc[1]['Value'] > np.mean(df[feature_imp.iloc[1]['Feature']]) else 'Low'} relative to average
        
        **Note**: This is a data-driven prediction based on machine learning analysis.
        For medical advice, please consult with a healthcare professional.
        """)
