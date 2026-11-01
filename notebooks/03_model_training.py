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

###################################
#Evaluate Model Performance
################################
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Performance:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

#3333333333333333333333333333
#Generate Confusion Matrix
#3333333333333333333333333333333
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


#Generate Classification Report

from sklearn.metrics import classification_report

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))



#Create Decision Tree Model

from sklearn.tree import DecisionTreeClassifier

decision_tree_model = DecisionTreeClassifier(
    random_state=42
)

#Train Decision Tree Model

decision_tree_model.fit(X_train, y_train)

print("\nDecision Tree Model Trained Successfully!")


#Make Predictions Using Decision Tree

y_pred_tree = decision_tree_model.predict(X_test)

print("\nDecision Tree Predictions Generated Successfully!")

print("\nFirst 10 Actual Values:")
print(y_test.head(10).tolist())

print("\nFirst 10 Decision Tree Predictions:")
print(y_pred_tree[:10])