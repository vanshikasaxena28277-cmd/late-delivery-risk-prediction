import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.title("Late Delivery Risk Prediction")

# Safe model loading
model_path = os.path.join("models", "model.pkl")

if not os.path.exists(model_path):
    st.error("Model file not found. Please upload model.pkl in models folder.")
else:
    model = pickle.load(open(model_path, "rb"))

    # Upload data
    file = st.file_uploader("Upload CSV")

    if file:
        df = pd.read_csv(file)

        st.write("Data Preview", df.head())

        if st.button("Predict Risk"):

            X = df.copy()

            # Drop target if exists
            if 'Late_delivery_risk' in X.columns:
                X = X.drop(['Late_delivery_risk'], axis=1)

            # Encode categorical columns
            cat_cols = X.select_dtypes(include=['object']).columns
            le = LabelEncoder()

            for col in cat_cols:
                X[col] = le.fit_transform(X[col].astype(str))

            # Scale
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            # Prediction
            predictions = model.predict(X)

            df['Predicted Risk'] = predictions

            st.write(df[['Predicted Risk']])

            high_risk = df[df['Predicted Risk'] == 1]

            st.write("High Risk Orders:", high_risk.shape[0])
