
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

# -----------------------------------------
# Churn Percentage by Contract Type Visualization
# -----------------------------------------

# Create the contract-wise churn bar chart.
ax = contract_churn_percentage[["Yes"]].plot(
    kind="bar",
    figsize=(8, 5),
    legend=False
)

# Add percentage labels above each bar.
for bar in ax.patches:
    height = bar.get_height()

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 1,
        f"{height:.2f}%",
        ha="center",
        va="bottom",
        fontsize=10
    )

# Add chart title and axis labels.
plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Churn Percentage")
plt.xticks(rotation=0)
plt.ylim(0, 50)
plt.tight_layout()

#plt.show()

# -----------------------------------------
# Customer Churn Analysis by Tenure
# -----------------------------------------

# Check the minimum and maximum customer tenure.
print("\nCustomer Tenure Statistics:")
print("Minimum tenure:", df["tenure"].min(), "months")
print("Maximum tenure:", df["tenure"].max(), "months")

# Group customers based on their tenure.
df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=["0-1 Year", "1-2 Years", "2-4 Years", "4-6 Years"]
)

# Display the number of customers in each tenure group.
print("\nCustomer Count by Tenure Group:")
print(df["TenureGroup"].value_counts().sort_index())

# Analyze customer churn across tenure groups.
tenure_churn = pd.crosstab(
    df["TenureGroup"],
    df["Churn"]
)

print("\nCustomer Churn by Tenure Group:")
print(tenure_churn)

# Calculate churn percentage within each tenure group.
tenure_churn_percentage = pd.crosstab(
    df["TenureGroup"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Tenure Group:")
print(tenure_churn_percentage.round(2))

# ----------------------------------------
# Visualization: Churn Percentage by Tenure
# ----------------------------------------

ax = tenure_churn_percentage[["Yes"]].plot(
    kind="bar",
    figsize=(9, 5),
    legend=False,
    color="coral",
    edgecolor="black",
    width=0.6
)

# Add percentage labels above each bar
for bar in ax.patches:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 1,
        f"{height:.2f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )

# Customize chart appearance
plt.title("Customer Churn Rate by Tenure Group", fontsize=14, fontweight="bold")
plt.xlabel("Tenure Group", fontsize=11)
plt.ylabel("Churn Percentage (%)", fontsize=11)
plt.ylim(0, 55)
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()

#plt.show()

# ----------------------------------------
# Analysis: Customer Churn by Payment Method
# ----------------------------------------

# Calculate customer counts by payment method and churn status
payment_churn = pd.crosstab(
    df["PaymentMethod"],
    df["Churn"]
)

print("\nCustomer Churn by Payment Method:")
print(payment_churn)

# Calculate churn percentage for each payment method
payment_churn_percentage = pd.crosstab(
    df["PaymentMethod"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Payment Method:")
print(payment_churn_percentage.round(2))

# ----------------------------------------
# Visualization: Churn Percentage by Payment Method
# ----------------------------------------

ax = payment_churn_percentage[["Yes"]].plot(
    kind="bar",
    figsize=(10, 5),
    legend=False,
    color="coral",
    edgecolor="black",
    width=0.6
)

# Add percentage labels above each bar
for bar in ax.patches:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 1,
        f"{height:.2f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )

# Customize chart appearance
plt.title("Customer Churn Rate by Payment Method",
          fontsize=14, fontweight="bold")
plt.xlabel("Payment Method", fontsize=11)
plt.ylabel("Churn Percentage (%)", fontsize=11)
plt.ylim(0, 55)
plt.xticks(rotation=20, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()


#plt.show()

#------------------------------------------------
#Analyze churn by customer demographics like:
#-------------------------------------------------
#   >Senior citizen status
#   >Gender
#   >Whether customers have partners
#   >Whether customers have dependents
#--------------------------------------------------


# ----------------------------------------
# Analysis: Customer Churn by Senior Citizen Status
# ----------------------------------------

# Calculate customer counts by senior citizen status and churn
senior_churn = pd.crosstab(
    df["SeniorCitizen"],
    df["Churn"]
)

print("\nCustomer Churn by Senior Citizen Status:")
print(senior_churn)

# Calculate churn percentages for each group
senior_churn_percentage = pd.crosstab(
    df["SeniorCitizen"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Senior Citizen Status:")
print(senior_churn_percentage.round(2))


# ----------------------------------------
# Visualization: Churn by Senior Citizen Status
# ----------------------------------------

ax = senior_churn_percentage[["Yes"]].plot(
    kind="bar",
    figsize=(8, 5),
    legend=False,
    color="coral",
    edgecolor="black",
    width=0.5
)

# Add percentage labels above each bar
for bar in ax.patches:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 1,
        f"{height:.2f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )

# Customize chart appearance
plt.title("Customer Churn Rate by Senior Citizen Status",
          fontsize=14, fontweight="bold")
plt.xlabel("Senior Citizen Status (0 = No, 1 = Yes)", fontsize=11)
plt.ylabel("Churn Percentage (%)", fontsize=11)
plt.ylim(0, 50)
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
#plt.show()


# ----------------------------------------
# Analysis: Customer Churn by Gender
# ----------------------------------------

# Calculate customer counts by gender and churn status
gender_churn = pd.crosstab(
    df["gender"],
    df["Churn"]
)

print("\nCustomer Churn by Gender:")
print(gender_churn)

# Calculate churn percentage for each gender
gender_churn_percentage = pd.crosstab(
    df["gender"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Gender:")
print(gender_churn_percentage.round(2))


# ----------------------------------------
# Visualization: Churn Percentage by Gender
# ----------------------------------------

ax = gender_churn_percentage[["Yes"]].plot(
    kind="bar",
    figsize=(8, 5),
    legend=False,
    color="coral",
    edgecolor="black",
    width=0.5
)

# Add percentage labels above each bar
for bar in ax.patches:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 1,
        f"{height:.2f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )

# Customize chart appearance
plt.title("Customer Churn Rate by Gender",
          fontsize=14, fontweight="bold")
plt.xlabel("Gender", fontsize=11)
plt.ylabel("Churn Percentage (%)", fontsize=11)
plt.ylim(0, 35)
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
#plt.show()


# ----------------------------------------
# Analysis: Customer Churn by Partner Status
# ----------------------------------------

# Calculate customer counts by partner status and churn
partner_churn = pd.crosstab(
    df["Partner"],
    df["Churn"]
)

print("\nCustomer Churn by Partner Status:")
print(partner_churn)

# Calculate churn percentages
partner_churn_percentage = pd.crosstab(
    df["Partner"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Partner Status:")
print(partner_churn_percentage.round(2))


# ----------------------------------------
# Visualization: Churn Percentage by Partner Status
# ----------------------------------------

ax = partner_churn_percentage[["Yes"]].plot(
    kind="bar",
    figsize=(8, 5),
    legend=False,
    color="coral",
    edgecolor="black",
    width=0.5
)

# Add percentage labels above each bar
for bar in ax.patches:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 1,
        f"{height:.2f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )

# Customize chart appearance
plt.title("Customer Churn Rate by Partner Status",
          fontsize=14, fontweight="bold")
plt.xlabel("Has a Partner", fontsize=11)
plt.ylabel("Churn Percentage (%)", fontsize=11)
plt.ylim(0, 40)
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
#plt.show()


# ----------------------------------------
# Analysis: Customer Churn by Dependent Status
# ----------------------------------------

# Calculate customer counts by dependent status and churn
dependents_churn = pd.crosstab(
    df["Dependents"],
    df["Churn"]
)

print("\nCustomer Churn by Dependent Status:")
print(dependents_churn)

# Calculate churn percentages
dependents_churn_percentage = pd.crosstab(
    df["Dependents"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Dependent Status:")
print(dependents_churn_percentage.round(2))


# ----------------------------------------
# Visualization: Churn Percentage by Dependent Status
# ----------------------------------------

ax = dependents_churn_percentage[["Yes"]].plot(
    kind="bar",
    figsize=(8, 5),
    legend=False,
    color="coral",
    edgecolor="black",
    width=0.5
)

# Add percentage labels above each bar
for bar in ax.patches:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 1,
        f"{height:.2f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )

# Customize chart appearance
plt.title("Customer Churn Rate by Dependent Status",
          fontsize=14, fontweight="bold")
plt.xlabel("Has Dependents", fontsize=11)
plt.ylabel("Churn Percentage (%)", fontsize=11)
plt.ylim(0, 40)
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
#plt.show()


# ----------------------------------------
# Analysis: Customer Churn by Internet Service
# ----------------------------------------

# Calculate customer counts by internet service and churn
internet_churn = pd.crosstab(
    df["InternetService"],
    df["Churn"]
)

print("\nCustomer Churn by Internet Service:")
print(internet_churn)

# Calculate churn percentages
internet_churn_percentage = pd.crosstab(
    df["InternetService"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Internet Service:")
print(internet_churn_percentage.round(2))


# ----------------------------------------
# Visualization: Churn Percentage by Internet Service
# ----------------------------------------

ax = internet_churn_percentage[["Yes"]].plot(
    kind="bar",
    figsize=(9, 5),
    legend=False,
    color="coral",
    edgecolor="black",
    width=0.6
)

# Add percentage labels above each bar
for bar in ax.patches:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 1,
        f"{height:.2f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )

# Customize chart appearance
plt.title("Customer Churn Rate by Internet Service",
          fontsize=14, fontweight="bold")
plt.xlabel("Internet Service", fontsize=11)
plt.ylabel("Churn Percentage (%)", fontsize=11)
plt.ylim(0, 50)
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
#plt.show()


# ----------------------------------------
# Analysis: Customer Churn by Online Security
# ----------------------------------------

# Calculate customer counts by online security and churn
security_churn = pd.crosstab(
    df["OnlineSecurity"],
    df["Churn"]
)

print("\nCustomer Churn by Online Security:")
print(security_churn)

# Calculate churn percentages
security_churn_percentage = pd.crosstab(
    df["OnlineSecurity"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Online Security:")
print(security_churn_percentage.round(2))

# ----------------------------------------
# Visualization: Churn by Online Security
# ----------------------------------------

security_churn_rate = security_churn_percentage["Yes"]

plt.figure(figsize=(9, 5))

bars = plt.bar(
    security_churn_rate.index,
    security_churn_rate.values,
    color="coral",
    edgecolor="black"
)

plt.title("Customer Churn Rate by Online Security")
plt.xlabel("Online Security")
plt.ylabel("Churn Rate (%)")
plt.ylim(0, 50)

# Display percentage labels above each bar
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 1,
        f"{height:.2f}%",
        ha="center"
    )

plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
#plt.show()


# ----------------------------------------
# Analysis: Customer Churn by Online Backup
# ----------------------------------------

# Calculate customer counts by online backup and churn
backup_churn = pd.crosstab(
    df["OnlineBackup"],
    df["Churn"]
)

print("\nCustomer Churn by Online Backup:")
print(backup_churn)

# Calculate churn percentages
backup_churn_percentage = pd.crosstab(
    df["OnlineBackup"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Online Backup:")
print(backup_churn_percentage.round(2))


# ----------------------------------------
# Visualization: Churn by Online Backup
# ----------------------------------------

backup_churn_rate = backup_churn_percentage["Yes"]

plt.figure(figsize=(9, 5))

bars = plt.bar(
    backup_churn_rate.index,
    backup_churn_rate.values,
    color="coral",
    edgecolor="black"
)

plt.title("Customer Churn Rate by Online Backup")
plt.xlabel("Online Backup")
plt.ylabel("Churn Rate (%)")
plt.ylim(0, 50)

# Display percentage labels above each bar
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 1,
        f"{height:.2f}%",
        ha="center"
    )

plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
#plt.show()


# ----------------------------------------
# Analysis: Customer Churn by Device Protection
# ----------------------------------------

# Calculate customer counts by device protection and churn
device_churn = pd.crosstab(
    df["DeviceProtection"],
    df["Churn"]
)

print("\nCustomer Churn by Device Protection:")
print(device_churn)

# Calculate churn percentages
device_churn_percentage = pd.crosstab(
    df["DeviceProtection"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Device Protection:")
print(device_churn_percentage.round(2))