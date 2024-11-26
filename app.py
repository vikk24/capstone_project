import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model and scaler
model = joblib.load('best_rf_model.pkl')
scaler = joblib.load('scaler.pkl')

# Create the app's UI
st.title('Car Selling Price Prediction')

# Input features from the user
year = st.number_input("Year", min_value=2000, max_value=2024, step=1)
km_driven = st.number_input("Kilometers Driven")
transmission = st.selectbox("Transmission", options=["Automatic", "Manual"])
owner = st.selectbox("Owner", options=["First Owner", "Second Owner", "Third Owner", "Fourth & Above"])
fuel_type = st.selectbox("Fuel Type", options=["Petrol", "Diesel", "Electric", "LPG"])
seller_type = st.selectbox("Seller Type", options=["Individual", "Trustmark Dealer"])

# Convert input values to a DataFrame (match your model's input format)
input_data = pd.DataFrame({
    'year': [year],
    'km_driven': [km_driven],
    'transmission': [transmission],
    'owner': [owner],
    'fuel_type': [fuel_type],
    'seller_type': [seller_type],
})

# Apply one-hot encoding on the input features
input_data_encoded = pd.get_dummies(input_data)

# Align the columns with the training data's columns
model_columns = joblib.load('model_columns.pkl')  # This should be saved during training with column names
input_data_encoded = input_data_encoded.reindex(columns=model_columns, fill_value=0)

# Add a button to trigger the prediction
if st.button('Get Predicted Selling Price'):
    # Apply the scaler to the input data
    input_data_scaled = scaler.transform(input_data_encoded)

    # Make prediction
    prediction = model.predict(input_data_scaled)

    # Show the prediction
    st.write(f"Predicted Selling Price: ₹{prediction[0]:,.2f}")
