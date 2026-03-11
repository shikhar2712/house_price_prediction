import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("models/house_price_model.pkl")

model_features = model.feature_names_in_

st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏡 House Price Prediction")

st.write(
    """
    This demo predicts house prices using a **Random Forest Regression model**
    trained on the Ames Housing dataset.
    """
)

st.markdown("---")

st.header("Enter Property Details")

col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider("Overall Quality", 1, 10, 5)
    gr_liv_area = st.number_input("Living Area (sq ft)", 500, 5000, 1500)
    first_flr = st.number_input("1st Floor Area (sq ft)", 400, 3000, 1200)
    garage_cars = st.slider("Garage Capacity", 0, 4, 2)

with col2:
    second_flr = st.number_input("2nd Floor Area (sq ft)", 0, 2000, 300)
    full_bath = st.slider("Full Bathrooms", 0, 4, 2)
    lot_area = st.number_input("Lot Area (sq ft)", 1000, 20000, 8000)
    kitchen_qual = st.slider("Kitchen Quality", 1, 5, 3)

# Build feature dataframe
input_df = pd.DataFrame(columns=model_features)
input_df.loc[0] = 0

features = {
    "Overall Qual": overall_qual,
    "Gr Liv Area": gr_liv_area,
    "1st Flr SF": first_flr,
    "2nd Flr SF": second_flr,
    "Garage Cars": garage_cars,
    "Full Bath": full_bath,
    "Lot Area": lot_area,
    "Kitchen Qual": kitchen_qual,
}

for feature, value in features.items():
    if feature in input_df.columns:
        input_df[feature] = value

if st.button("Predict House Price"):

    prediction = model.predict(input_df)[0]

    st.success(f"💰 Estimated House Price: **${prediction:,.0f}**")

    # Simple confidence estimate
    tree_preds = [tree.predict(input_df.values)[0] for tree in model.estimators_]
    confidence = pd.Series(tree_preds).std()

    st.info(f"Prediction variability (model uncertainty): ± ${confidence:,.0f}")

st.markdown("---")

st.header("Model Insights")

# Feature importance
importance = pd.Series(
    model.feature_importances_,
    index=model_features
).sort_values(ascending=False).head(10)

fig, ax = plt.subplots()
importance.sort_values().plot.barh(ax=ax)
ax.set_title("Top Features Influencing Price")
st.pyplot(fig)

st.markdown("---")

st.caption(
    """
    Model: Random Forest Regressor  
    Dataset: Ames Housing  
    Test R² Score: ~0.91
    """
)