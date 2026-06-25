# ============================================================
# SALES PREDICTION USING PYTHON
# Dataset: Advertising.csv
# ============================================================

# =========================
# IMPORT LIBRARIES
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("Advertising.csv")

print("=" * 60)
print("SALES PREDICTION PROJECT")
print("=" * 60)

# =========================
# DATA OVERVIEW
# =========================

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

# =========================
# DATA CLEANING
# =========================

if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

print("\nDataset After Cleaning:")
print(df.head())

# =========================
# STATISTICAL SUMMARY
# =========================

print("\nStatistical Summary:")
print(df.describe())

# =========================
# CORRELATION HEATMAP
# =========================

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

# =========================
# PAIRPLOT
# =========================

pairplot = sns.pairplot(df)
pairplot.fig.suptitle("Relationship Between Variables", y=1.02)
pairplot.savefig("pairplot.png")
plt.close()

# =========================
# TV VS SALES
# =========================

plt.figure(figsize=(6,4))
sns.scatterplot(x="TV", y="Sales", data=df)
plt.title("TV Advertising vs Sales")
plt.tight_layout()
plt.savefig("tv_vs_sales.png")
plt.close()

# =========================
# RADIO VS SALES
# =========================

plt.figure(figsize=(6,4))
sns.scatterplot(x="Radio", y="Sales", data=df)
plt.title("Radio Advertising vs Sales")
plt.tight_layout()
plt.savefig("radio_vs_sales.png")
plt.close()

# =========================
# NEWSPAPER VS SALES
# =========================

plt.figure(figsize=(6,4))
sns.scatterplot(x="Newspaper", y="Sales", data=df)
plt.title("Newspaper Advertising vs Sales")
plt.tight_layout()
plt.savefig("newspaper_vs_sales.png")
plt.close()

# =========================
# SALES DISTRIBUTION
# =========================

plt.figure(figsize=(6,4))
sns.histplot(df["Sales"], bins=20, kde=True)
plt.title("Sales Distribution")
plt.tight_layout()
plt.savefig("sales_distribution.png")
plt.close()

# =========================
# FEATURE SELECTION
# =========================

X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# MODEL TRAINING
# =========================

model = LinearRegression()
model.fit(X_train, y_train)

# =========================
# PREDICTIONS
# =========================

y_pred = model.predict(X_test)

# =========================
# MODEL EVALUATION
# =========================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print("MAE :", round(mae, 2))
print("MSE :", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R² Score:", round(r2, 2))

# =========================
# ACTUAL VS PREDICTED TABLE
# =========================

results = pd.DataFrame({
    "Actual Sales": y_test,
    "Predicted Sales": y_pred
})

print("\nSample Predictions:")
print(results.head(10))

# =========================
# ACTUAL VS PREDICTED GRAPH
# =========================

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.close()

# =========================
# FEATURE IMPORTANCE
# =========================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nAdvertising Impact:")
print(importance)

# =========================
# FEATURE IMPORTANCE GRAPH
# =========================

plt.figure(figsize=(6,4))
sns.barplot(x="Feature", y="Coefficient", data=importance)
plt.title("Advertising Impact on Sales")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# =========================
# FUTURE SALES PREDICTION
# =========================

future_data = pd.DataFrame({
    "TV": [250],
    "Radio": [40],
    "Newspaper": [50]
})

future_sales = model.predict(future_data)

print("\nPredicted Future Sales:")
print(round(future_sales[0], 2))

# =========================
# SAVE RESULTS
# =========================

results.to_csv("sales_predictions.csv", index=False)

print("\nGenerated Files:")
print("images/correlation_heatmap.png")
print("images/pairplot.png")
print("images/tv_vs_sales.png")
print("images/radio_vs_sales.png")
print("images/newspaper_vs_sales.png")
print("images/sales_distribution.png")
print("images/actual_vs_predicted.png")
print("images/feature_importance.png")
print("sales_predictions.csv")