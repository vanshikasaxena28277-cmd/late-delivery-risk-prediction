import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    df = df.dropna()

    # Target
    y = df['Late_delivery_risk']
    X = df.drop(['Late_delivery_risk'], axis=1)

    # Encode categorical
    cat_cols = X.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in cat_cols:
        X[col] = le.fit_transform(X[col].astype(str))

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y
