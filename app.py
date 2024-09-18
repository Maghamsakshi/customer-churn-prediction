import pandas as pd
import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

#load dataset
dh=pd.read_csv("Customertravel.csv")



le = LabelEncoder()
dh['FrequentFlyer'] = le.fit_transform(dh['FrequentFlyer'])
dh['AnnualIncomeClass'] = le.fit_transform(dh['AnnualIncomeClass'])
dh['AccountSyncedToSocialMedia'] = le.fit_transform(dh['AccountSyncedToSocialMedia'])
dh['BookedHotelOrNot'] = le.fit_transform(dh['BookedHotelOrNot'])






# Title of the app
st.title("Customer Churn Prediction")




st.write("""### ENTER THE CUSTOMER DETAILS : ###""")

age = st.number_input("Enter your Age")
ServicesOpted = st.slider("ServicesOpted",0,6)
FrequentFlyer = st.selectbox("FrequentFlyer",["Yes","No","No record"])
AnnualIncomeClass = st.selectbox("AnnualIncomeClass",["High Income","Middle Income","Low Income"])
AccountSyncedToSocialMedia = st.selectbox("AccountSyncedToSocialMedia",["Yes","No"])
BookedHotelOrNot = st.selectbox("BookedHotel",["Yes","No","No record"])



ok = st.button("Predict")

# Prepare input data
if ok:
    # Ensure you handle the case where the button is clicked
    dh = pd.DataFrame({
        'age': [age],
        'ServicesOpted': [ServicesOpted],
        "FrequentFlyer": [FrequentFlyer],
        "AnnualIncomeClass": [AnnualIncomeClass],
        "AccountSyncedToSocialMedia": [AccountSyncedToSocialMedia],
        "BookedHotelOrNot": [BookedHotelOrNot]
    })

    

    # Map categorical values to numerical values
    dh['AnnualIncomeClass'] = dh['AnnualIncomeClass'].map({"High Income": 0, "Middle Income": 2, "Low Income": 1})
    dh['FrequentFlyer'] = dh['FrequentFlyer'].map({'Yes': 2, 'No': 0, 'No record': 1})
    dh['AccountSyncedToSocialMedia'] = dh['AccountSyncedToSocialMedia'].map({'Yes': 1, 'No': 0})
    dh['BookedHotelOrNot'] = dh['BookedHotelOrNot'].map({'Yes': 1, 'No': 0, 'No record': 0})

    # Load the model
    import pickle
    with open('churn_prediction_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)['model']

    # Ensure all necessary fields are numerical before prediction

        
# Make prediction
    prediction = loaded_model.predict(dh)
    result = "Churn" if prediction[0] == 1 else "Not Churn"
    st.write("Prediction:", result)
