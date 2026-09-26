import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('logi.sav')

st.title('Delivery Delay Prediction')
st.write('Enter the features below to predict if there will be a delivery delay.')

# Input fields for features
delivery_distance = st.number_input('Delivery Distance (e.g., 20.0)', value=20.0, format="%.2f")
traffic_congestion = st.slider('Traffic Congestion (1=Low, 5=High)', 1, 5, 3)
weather_condition = st.slider('Weather Condition (1=Good, 5=Bad)', 1, 5, 2)
delivery_slot = st.slider('Delivery Slot (1, 2, or 3)', 1, 3, 2)
driver_experience = st.number_input('Driver Experience (Years, e.g., 5)', value=5, min_value=0, max_value=30)
num_stops = st.number_input('Number of Stops (e.g., 2)', value=2, min_value=0, max_value=20)
vehicle_age = st.number_input('Vehicle Age (Years, e.g., 3)', value=3, min_value=0, max_value=20)
road_condition_score = st.slider('Road Condition Score (1=Bad, 5=Good)', 1, 5, 3)
package_weight = st.number_input('Package Weight (kg, e.g., 12.0)', value=12.0, format="%.2f")
fuel_efficiency = st.number_input('Fuel Efficiency (km/l, e.g., 12.0)', value=12.0, format="%.2f")
warehouse_processing_time = st.number_input('Warehouse Processing Time (minutes, e.g., 120)', value=120, min_value=0)

# Create a DataFrame from user inputs
input_data = pd.DataFrame([{
    'Delivery_Distance': delivery_distance,
    'Traffic_Congestion': traffic_congestion,
    'Weather_Condition': weather_condition,
    'Delivery_Slot': delivery_slot,
    'Driver_Experience': driver_experience,
    'Num_Stops': num_stops,
    'Vehicle_Age': vehicle_age,
    'Road_Condition_Score': road_condition_score,
    'Package_Weight': package_weight,
    'Fuel_Efficiency': fuel_efficiency,
    'Warehouse_Processing_Time': warehouse_processing_time
}])

if st.button('Predict Delay'):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.error('Prediction: \"Delay Expected\"')
    else:
        st.success('Prediction: \"No Delay Expected\"')
    
    st.write(f"Probability of No Delay: {prediction_proba[0][0]*100:.2f}%")
    st.write(f"Probability of Delay: {prediction_proba[0][1]*100:.2f}%")


st.write("\n--- Make sure to adjust values if warnings about feature names arise. --- ")
