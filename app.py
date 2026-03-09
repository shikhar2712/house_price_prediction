import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/house_price_model.pkl")

# Get features used during training
model_features = model.feature_names_in_

st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏡 House Price Prediction")
st.write(
"""
This application predicts house prices using a **Random Forest Machine Learning model**  
trained on the Ames Housing dataset.
"""
)

st.header("Enter Property Details")

# Key features based on your feature importance
overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)

gr_liv_area = st.number_input(
    "Above Ground Living Area (sq ft)", min_value=500, max_value=5000, value=1500
)

first_flr = st.number_input(
    "First Floor Area (sq ft)", min_value=400, max_value=3000, value=1200
)

second_flr = st.number_input(
    "Second Floor Area (sq ft)", min_value=0, max_value=2000, value=300
)

garage_cars = st.slider("Garage Capacity (cars)", 0, 4, 2)

full_bath = st.slider("Full Bathrooms", 0, 4, 2)

lot_area = st.number_input(
    "Lot Area (sq ft)", min_value=1000, max_value=20000, value=8000
)

kitchen_qual = st.slider("Kitchen Quality (1-5)", 1, 5, 3)

# Build full feature dataframe
input_df = pd.DataFrame(columns=model_features)
input_df.loc[0] = 0

# Populate known important features
feature_values = {
    "Overall Qual": overall_qual,
    "Gr Liv Area": gr_liv_area,
    "1st Flr SF": first_flr,
    "2nd Flr SF": second_flr,
    "Garage Cars": garage_cars,
    "Full Bath": full_bath,
    "Lot Area": lot_area,
    "Kitchen Qual": kitchen_qual,
}

for feature, value in feature_values.items():
    if feature in input_df.columns:
        input_df[feature] = value


# Prediction
if st.button("Predict House Price"):

    prediction = model.predict(input_df)[0]

    st.success(f"💰 Estimated House Price: **${prediction:,.2f}**")

st.markdown("---")

st.caption(
"Model: Random Forest Regressor | Dataset: Ames Housing | R² ≈ 0.91"
)