# ----------------------------------------
# Customer Churn Prediction
# Data Preprocessing
# ----------------------------------------

# Import required libraries
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

# Clean TotalCharges
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].str.strip(),
    errors="coerce"
)

# Handle missing TotalCharges for zero-tenure customers
df.loc[df["tenure"] == 0, "TotalCharges"] = (
    df.loc[df["tenure"] == 0, "TotalCharges"].fillna(0)
)

# Verify the dataset
print("Dataset Shape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum().sum())


# ----------------------------------------
# Feature and Target Separation
# ----------------------------------------

# Remove customerID because it is only an identifier
# Remove TenureGroup because it was created only for exploration
ml_df = df.drop(columns=["customerID"])


# Remove Churn because it is the target variable
X = ml_df.drop(columns=["Churn"])
y = ml_df["Churn"]

# Display dataset shapes
print("\nFeature Dataset Shape:", X.shape)
print("Target Dataset Shape:", y.shape)

# ----------------------------------------
# Step 3: Identify Categorical Features
# ----------------------------------------

# Select columns containing text data
categorical_columns = X.select_dtypes(
    include=["str", "object"]
).columns

# Display categorical column names
print("\nCategorical Columns:")
print(categorical_columns.tolist())

# Display the total number of categorical columns
print("\nNumber of Categorical Columns:", len(categorical_columns))