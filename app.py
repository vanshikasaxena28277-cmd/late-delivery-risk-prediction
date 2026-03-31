import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder

st.title("Late Delivery Risk Prediction")

# Load files safely
model_path = "best_model.pkl"
scaler_path = "scaler.pkl"
features_path = "feature_names.pkl"

if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
    st.error("Model or scaler file missing. Please upload all .pkl files.")
else:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(features_path)

    file = st.file_uploader("Upload CSV")

    if file:
        df = pd.read_csv(file)

        st.write("Data Preview", df.head())

        if st.button("Predict Risk"):

            X = df.copy()

            # Drop target if present
            if 'Late_delivery_risk' in X.columns:
                X = X.drop(columns=['Late_delivery_risk'])

            # Encode categorical columns
            cat_cols = X.select_dtypes(include='object').columns
            le = LabelEncoder()

            for col in cat_cols:
                X[col] = le.fit_transform(X[col].astype(str))

            # Match training features
            X = X.reindex(columns=feature_names, fill_value=0)

            # Scale
            X_scaled = scaler.transform(X)

            # Predict
            predictions = model.predict(X_scaled)

            df['Predicted Risk'] = predictions

            st.write(df[['Predicted Risk']])

            high_risk = df[df['Predicted Risk'] == 1]
            st.write("High Risk Orders:", high_risk.shape[0])
