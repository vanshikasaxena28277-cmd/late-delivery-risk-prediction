import streamlit as st
import pandas as pd
import pickle

st.title("Late Delivery Risk Prediction")

# Load model
model = pickle.load(open("models/model.pkl", "rb"))

# Upload data
file = st.file_uploader("Upload CSV")

if file:
    df = pd.read_csv(file)

    st.write("Data Preview", df.head())

    if st.button("Predict Risk"):
        X = df.drop(['Late_delivery_risk'], axis=1, errors='ignore')

        predictions = model.predict(X)

        df['Predicted Risk'] = predictions

        st.write(df[['Predicted Risk']])

        high_risk = df[df['Predicted Risk'] == 1]

        st.write("High Risk Orders:", high_risk.shape[0])
