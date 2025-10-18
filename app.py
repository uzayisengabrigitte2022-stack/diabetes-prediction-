import numpy as np
import pickle
import pandas as pd
import streamlit as st

# Load the saved model
loaded_model = pickle.load(open('C:/Users/user/Desktop/diabete/diabetes_model.sav', 'rb'))

# Prediction function
def diabete_data_prediction(weight, stress_level, blood_glucose, user_id,
                            physical_activity, diet, medication_adherence,
                            sleep_hours, hydration_level, bmi):
    
    # Define features in the exact order as during model training
    feature_names = [
        'weight', 'stress_level', 'blood_glucose', 'user_id',
        'physical_activity', 'diet', 'medication_adherence',
        'sleep_hours', 'hydration_level', 'bmi'
    ]
    
    # Create DataFrame from input values
    new_data = pd.DataFrame([[weight, stress_level, blood_glucose, user_id,
                              physical_activity, diet, medication_adherence,
                              sleep_hours, hydration_level, bmi]],
                            columns=feature_names)
    
    # Make prediction
    predicted_diabete = loaded_model.predict(new_data)
    
    return predicted_diabete[0]


# Streamlit app
def main():
    st.title("🩺 Diabetes Prediction App")
    st.write("Enter patient details to predict diabetes risk:")

    # Input fields
    weight = st.number_input("Weight (kg)", min_value=0.0, max_value=300.0, value=77.7, step=0.1)
    stress_level = st.number_input("Stress Level (1-10)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    blood_glucose = st.number_input("Blood Glucose Level (mg/dL)", min_value=0.0, max_value=500.0, value=186.0, step=0.1)
    user_id = st.number_input("User ID", min_value=0, max_value=1000, value=1, step=1)
    physical_activity = st.number_input("Physical Activity (0=No,1=Yes)", min_value=0, max_value=1, value=0, step=1)
    diet = st.number_input("Diet (0=No,1=Yes)", min_value=0, max_value=1, value=1, step=1)
    medication_adherence = st.number_input("Medication Adherence (0=No,1=Yes)", min_value=0, max_value=1, value=0, step=1)
    sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=6.33, step=0.1)
    hydration_level = st.number_input("Hydration Level (0=Low,1=Good)", min_value=0, max_value=1, value=1, step=1)
    bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, value=22.0, step=0.1)

    # Predict button
    if st.button("Predict Diabetes"):
        # Call prediction function
        prediction = diabete_data_prediction(weight, stress_level, blood_glucose, user_id,
                                             physical_activity, diet, medication_adherence,
                                             sleep_hours, hydration_level, bmi)
        
        # Display result
        if prediction == 1 or prediction == 'Diabetes':
            st.error("🚨 The patient is predicted to have **Diabetes**.")
        else:
            st.success("✅ The patient is predicted to **NOT have Diabetes**.")


if __name__ == "__main__":
    main()
