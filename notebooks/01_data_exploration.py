
import pandas as pd
import matplotlib.pyplot as plt

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

# Replace missing TotalCharges with 0 for customers
# who have zero months of service.
df.loc[df["tenure"] == 0, "TotalCharges"] = (
    df.loc[df["tenure"] == 0, "TotalCharges"].fillna(0)
)

# Verify that missing values have been handled.
print("\nMissing TotalCharges after cleaning:",
      df["TotalCharges"].isnull().sum())

# Verify TotalCharges for customers with zero tenure.
zero_tenure_customers = df[df["tenure"] == 0]

print("\nCustomers with zero tenure:",
      len(zero_tenure_customers))

print("\nTotalCharges for zero-tenure customers:")
print(zero_tenure_customers["TotalCharges"])

# df.info() displays the original data types, not the cleaned data types.
#display the updated information after cleaning of data.
# Check the dataset structure after data cleaning.
print("\nDataset information after cleaning:")
df.info()

# Verify the total number of missing values.
print("\nTotal missing values after cleaning:",
      df.isnull().sum().sum())

# Verify the dataset dimensions remain unchanged.
print("\nDataset shape after cleaning:", df.shape)

# -----------------------------------------
# Exploratory Data Analysis (EDA)
# -----------------------------------------

# Analyze the distribution of customer churn.
print("\nCustomer Churn Distribution:")
print(df["Churn"].value_counts())

# Calculate the percentage of customers in each churn category.
churn_percentage = df["Churn"].value_counts(normalize=True) * 100

print("\nCustomer Churn Percentage:")
print(churn_percentage.round(2))

# Visualize the customer churn distribution.
#churn_counts = df["Churn"].value_counts()

#churn_counts.plot(kind="bar", figsize=(7, 5))

#plt.title("Customer Churn Distribution")
#plt.xlabel("Customer Churn")
#plt.ylabel("Number of Customers")
#plt.xticks(rotation=0)
#plt.tight_layout()
#plt.show()

# -----------------------------------------
# Improved Churn Distribution Visualization
# -----------------------------------------

# Create a separate bar chart with customer counts.
churn_counts = df["Churn"].value_counts()

fig, ax = plt.subplots(figsize=(7, 5))

bars = ax.bar(churn_counts.index, churn_counts.values)

# Add customer counts above each bar.
for bar in bars:
    height = bar.get_height()

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        str(int(height)),
        ha="center",
        va="bottom"
    )

ax.set_title("Customer Churn Distribution with Counts")
ax.set_xlabel("Customer Churn")
ax.set_ylabel("Number of Customers")

plt.tight_layout()
#plt.show()

# -----------------------------------------
# Customer Churn Analysis by Contract Type
# -----------------------------------------

# Count customers by contract type and churn status.
contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"]
)

print("\nCustomer Churn by Contract Type:")
print(contract_churn)

# Calculate churn percentage within each contract type.
contract_churn_percentage = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Contract Type:")
print(contract_churn_percentage.round(2))