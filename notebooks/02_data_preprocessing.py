# ----------------------------------------
# Customer Churn Prediction
# Data Preprocessing
# ----------------------------------------

# Import required libraries
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

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

#from sklearn.preprocessing import OneHotEncoder

# Create the One-Hot Encoder
#encoder = OneHotEncoder(
 #   handle_unknown="ignore",
  #  sparse_output=False
#)

# Convert categorical values into numerical columns
#encoded_array = encoder.fit_transform(X[categorical_columns])

# Convert the encoded array into a DataFrame
#encoded_df = pd.DataFrame(
 #   encoded_array,
  #  columns=encoder.get_feature_names_out(categorical_columns),
   # index=X.index
#)

# Combine numerical and encoded categorical features
#X_encoded = pd.concat(
 #   [
  #      X.drop(columns=categorical_columns),
#        encoded_df
 #   ],
  #  axis=1
#)

# Display the encoded dataset information
#print("\nEncoded Feature Dataset Shape:", X_encoded.shape)

#print("\nFirst Five Rows of Encoded Dataset:")
#print(X_encoded.head())


#----------------------------------------
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




# ----------------------------------------0----
#Spliting Data into Training and Testing Sets
# ----------------------------------------------

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

# Display the shapes of the resulting datasets
print("\nTraining Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)

print("\nTraining Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)


 

# ----------------------------------------
# One-Hot Encoding Without Data Leakage
# ----------------------------------------

# Create the One-Hot Encoder
encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

# Identify categorical columns in the training data
categorical_columns = X_train.select_dtypes(
    include=["str", "object"]
).columns

# Fit the encoder using training data only
X_train_encoded_array = encoder.fit_transform(
    X_train[categorical_columns]
)

# Transform testing data using the same encoder
X_test_encoded_array = encoder.transform(
    X_test[categorical_columns]
)

# Convert training encoded data into a DataFrame
X_train_encoded = pd.DataFrame(
    X_train_encoded_array,
    columns=encoder.get_feature_names_out(categorical_columns),
    index=X_train.index
)

# Convert testing encoded data into a DataFrame
X_test_encoded = pd.DataFrame(
    X_test_encoded_array,
    columns=encoder.get_feature_names_out(categorical_columns),
    index=X_test.index
)

# Combine numerical and encoded categorical features
X_train_final = pd.concat(
    [
        X_train.drop(columns=categorical_columns),
        X_train_encoded
    ],
    axis=1
)

X_test_final = pd.concat(
    [
        X_test.drop(columns=categorical_columns),
        X_test_encoded
    ],
    axis=1
)

# Display final dataset shapes
print("\nFinal Training Dataset Shape:", X_train_final.shape)
print("Final Testing Dataset Shape:", X_test_final.shape)



#


