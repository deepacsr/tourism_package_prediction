import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

import streamlit as st

st.title("Test App Running")
st.write("If you see this, Docker is fine.")

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
PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])
OwnCar = st.selectbox("Own Car", ["Yes", "No"])
Passport = st.selectbox("Passport", ["Yes", "No"])

Age = st.number_input("Age", min_value=18, max_value=70, value=30, step =2)
DurationOfPitch = st.number_input("Duration Of Pitch(in Minutes)",min_value=5, max_value=127, value=15, step =5)
NumberOfPersonVisiting = st.number_input("Number Of Person Visiting",min_value=1, max_value=5, value=3, step =1)
NumberOfFollowups = st.number_input("Number of Followups",min_value=1, max_value=6, value=2, step =1)
NumberOfTrips = st.number_input("Number of Trips",min_value=1, max_value=22, value=4, step =1)
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting",min_value=0, max_value=3, value=2, step=1)
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score",min_value=1, max_value=5, value=3, step=1)
MonthlyIncome = st.number_input("Monthly Income",min_value=100, max_value=100000,value=25000,step = 3000)



# Assemble input into DataFrame
---
input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": 1 if Passport == "Yes" else 0,
    "OwnCar": 1 if OwnCar == "Yes" else 0,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "ProductPitched": ProductPitched,
    "NumberOfFollowups": NumberOfFollowups,
    "DurationOfPitch": DurationOfPitch
}])


# Predict button
if st.button("Predict Revenue"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result:")
    st.success(f"Customer will buy: **${prediction:,.2f} ")
