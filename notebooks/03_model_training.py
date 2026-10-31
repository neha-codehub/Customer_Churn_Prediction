#Import Required Libraries

import pandas as pd
from sklearn.linear_model import LogisticRegression


#Load Processed Datasets

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()


#Verify Loaded Datasets

print("Training Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)

print("Training Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)

#Create Logistic Regression Model

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


#Train the Model

logistic_model.fit(X_train, y_train)

print("\nLogistic Regression Model Trained Successfully!")


#Make Predictions

y_pred = logistic_model.predict(X_test)

print("\nPredictions Generated Successfully!")

print("\nFirst 10 Actual Values:")
print(y_test.head(10).tolist())

print("\nFirst 10 Predicted Values:")
print(y_pred[:10])