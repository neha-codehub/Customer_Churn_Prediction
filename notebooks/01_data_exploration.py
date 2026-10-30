
import pandas as pd

# Load the original customer churn dataset.
df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

# Check the dataset dimensions and structure.
print("Dataset shape:", df.shape)
df.info()

# Check for standard missing values in each column.
print("\nMissing values by column:")
print(df.isnull().sum())

# Check for blank strings in TotalCharges.
# isnull() alone does not detect empty strings.
blank_total_charges = (df["TotalCharges"].str.strip() == "").sum()
print("\nBlank TotalCharges entries:", blank_total_charges)

# Convert TotalCharges from text to numeric values.
# Invalid or blank entries will become NaN.
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].str.strip(),
    errors="coerce"
)

# Check the data type after conversion.
print("\nTotalCharges data type:", df["TotalCharges"].dtype)

# Count missing values created during conversion.
print("Missing TotalCharges after conversion:",
      df["TotalCharges"].isnull().sum())