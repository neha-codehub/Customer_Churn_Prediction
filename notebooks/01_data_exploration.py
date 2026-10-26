
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
