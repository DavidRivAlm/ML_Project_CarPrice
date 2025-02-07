import streamlit as st
import pickle
import pandas as pd

# Load the trained model
model = pickle.load(open('car_price_model.pkl', 'rb'))

st.title('Car Price Prediction App')
st.write('This web app predicts the **Price** of a car based on its features.')

# User input fields
brand = st.number_input('Brand (Encoded)', min_value=0, step=1)
model_type = st.number_input('Model (Encoded)', min_value=0, step=1)
fuel_type = st.number_input('Fuel Type (Encoded)', min_value=0, step=1)
transmission = st.number_input('Transmission (Encoded)', min_value=0, step=1)
mileage = st.number_input('Mileage')
doors = st.number_input('Number of Doors', min_value=2, max_value=5, step=1)
engine_size = st.number_input('Engine Size')
owner_count = st.number_input('Number of Previous Owners', min_value=0, step=1)
year = st.number_input('Year of Manufacture', min_value=1990, step=1)

# Convert input to DataFrame
user_data = pd.DataFrame({'Brand': [brand],
                          'Model': [model_type],
                          'Fuel_Type': [fuel_type],
                          'Transmission': [transmission],
                          'Mileage': [mileage],
                          'Doors': [doors],
                          'Engine_Size': [engine_size],
                          'Owner_Count': [owner_count],
                          'Year': [year]})

# Ensure column order matches model's training
expected_features = model.feature_names_in_
user_data = user_data[expected_features]

# Predict the car price
if st.button('Predict'):
    prediction = model.predict(user_data)
    st.write(f'The predicted car price is ${prediction[0]:,.2f}')

