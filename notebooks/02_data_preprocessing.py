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
#  Identify Categorical Features
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



# --------------------------------------------------------------------------=--------------------------------------
#One-Hot Encoding-for learning and checking the output
#Before model evaluation we'll adjust the workflow to fit the encoder only on training data to avoid data leakage.
# ----------------------------------------=-------------------------------------------------------------------------

from sklearn.preprocessing import OneHotEncoder

# Create the One-Hot Encoder
encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

# Convert categorical values into numerical columns
encoded_array = encoder.fit_transform(X[categorical_columns])

# Convert the encoded array into a DataFrame
encoded_df = pd.DataFrame(
    encoded_array,
    columns=encoder.get_feature_names_out(categorical_columns),
    index=X.index
)

# Combine numerical and encoded categorical features
X_encoded = pd.concat(
    [
        X.drop(columns=categorical_columns),
        encoded_df
    ],
    axis=1
)

# Display the encoded dataset information
print("\nEncoded Feature Dataset Shape:", X_encoded.shape)

print("\nFirst Five Rows of Encoded Dataset:")
print(X_encoded.head())


# ----------------------------------------
# Encode Target Variables
# ----------------------------------------

# Convert churn labels into numerical values
y_encoded = y.map({
    "No": 0,
    "Yes": 1
})

# Display target variable information
print("\nEncoded Target Values:")
print(y_encoded.value_counts())

print("\nTarget Data Type:", y_encoded.dtype)