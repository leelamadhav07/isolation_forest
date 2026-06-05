from sklearn.datasets import load_iris

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import IsolationForest

import pandas as pd

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
# ISOLATION FOREST
# --------------------------------

model = IsolationForest(contamination=0.05, random_state=42)

predictions = model.fit_predict(X_scaled)

# --------------------------------
# RESULTS
# --------------------------------

df = pd.DataFrame(X)

df["Anomaly"] = predictions

print(df.head())

print("Number of Anomalies:", sum(predictions == -1))
