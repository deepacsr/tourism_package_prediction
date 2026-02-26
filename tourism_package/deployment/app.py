import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(repo_id="deepacsr/tourism-package-prediction", filename="best_package_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Tourism package Prediction")
st.write("""
This application predicts if Customer is likely to purchase newly introduced  Wellness tourism package.
""")

# User input
TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
Occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business","Large Business"])
Gender = st.selectbox("Gender", ["Male", "Female"])
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe","Super Deluxe","King"])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager","AVP","VP"])
CityTier = st.selectbox("City Tier", [1, 2, 3])

Age = st.number_input("Age", min_value=18, max_value=70, value=30, step =2)
DurationOfPitch = st.number_input("Duration Of Pitch(in Minutes)",min_value=5, max_value=127, value=15, step =5)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'app_category': app_category,
    'free_or_paid': free_or_paid,
    'content_rating': content_rating,
    'screentime_category': screentime_category,
    'app_size_in_mb': app_size,
    'price_in_usd': price,
    'number_of_installs': installs,
    'average_screen_time': screen_time,
    'active_users': active_users,
    'no_of_short_ads_per_hour': short_ads,
    'no_of_long_ads_per_hour': long_ads
}])

# Predict button
if st.button("Predict Revenue"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result:")
    st.success(f"Estimated Ad Revenue: **${prediction:,.2f} USD**")
