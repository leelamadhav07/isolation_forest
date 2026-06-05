import streamlit as st

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import IsolationForest

# --------------------------------
# LOAD DATA
# --------------------------------

iris = load_iris()

X = iris.data

# --------------------------------
# SCALING
# --------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# --------------------------------
# PAGE
# --------------------------------

st.title("Isolation Forest")

st.write("Anomaly Detection")

# --------------------------------
# HYPERPARAMETER
# --------------------------------

contamination = st.slider("Contamination", 0.01, 0.20, 0.05)

# --------------------------------
# MODEL
# --------------------------------

model = IsolationForest(contamination=contamination, random_state=42)

predictions = model.fit_predict(X_scaled)

# --------------------------------
# COUNT
# --------------------------------

anomalies = sum(predictions == -1)

st.write(f"Detected Anomalies: {anomalies}")

# --------------------------------
# VISUALIZATION
# --------------------------------

fig, ax = plt.subplots()

ax.scatter(X[:, 0], X[:, 1], c=predictions)

ax.set_xlabel("Feature 1")

ax.set_ylabel("Feature 2")

ax.set_title("Anomaly Detection")

st.pyplot(fig)

# --------------------------------
# SHOW DATA
# --------------------------------

if st.checkbox("Show Dataset"):
    df = pd.DataFrame(X)
    df["Anomaly"] = predictions

    st.dataframe(df.head())
