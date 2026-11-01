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


#Evaluate Decision Tree Performance

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy_tree = accuracy_score(y_test, y_pred_tree)
precision_tree = precision_score(y_test, y_pred_tree)
recall_tree = recall_score(y_test, y_pred_tree)
f1_tree = f1_score(y_test, y_pred_tree)

print("\nDecision Tree Model Performance:")
print("Accuracy:", accuracy_tree)
print("Precision:", precision_tree)
print("Recall:", recall_tree)
print("F1 Score:", f1_tree)


#Create Random Forest Model

from sklearn.ensemble import RandomForestClassifier

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

#Train Random Forest Model

random_forest_model.fit(X_train, y_train)

print("\nRandom Forest Model Trained Successfully!")



#Make Predictions Using Random Forest

y_pred_rf = random_forest_model.predict(X_test)

print("\nRandom Forest Predictions Generated Successfully!")

print("\nFirst 10 Actual Values:")
print(y_test.head(10).tolist())

print("\nFirst 10 Random Forest Predictions:")
print(y_pred_rf[:10])



#Evaluate Random Forest Performance

accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print("\nRandom Forest Model Performance:")
print("Accuracy:", accuracy_rf)
print("Precision:", precision_rf)
print("Recall:", recall_rf)
print("F1 Score:", f1_rf)


#Create Balanced Logistic Regression Model

balanced_logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight="balanced"
)

# Train Balanced Logistic Regression Model

balanced_logistic_model.fit(X_train, y_train)

print("\nBalanced Logistic Regression Model Trained Successfully!")


#Make Predictions Using Balanced Logistic Regression

y_pred_balanced = balanced_logistic_model.predict(X_test)

print("\nBalanced Logistic Regression Predictions Generated Successfully!")



#Evaluate Balanced Logistic Regression Performance

accuracy_balanced = accuracy_score(y_test, y_pred_balanced)
precision_balanced = precision_score(y_test, y_pred_balanced)
recall_balanced = recall_score(y_test, y_pred_balanced)
f1_balanced = f1_score(y_test, y_pred_balanced)

print("\nBalanced Logistic Regression Performance:")
print("Accuracy:", accuracy_balanced)
print("Precision:", precision_balanced)
print("Recall:", recall_balanced)
print("F1 Score:", f1_balanced)


#Confusion Matrix for Balanced Logistic Regression

cm_balanced = confusion_matrix(y_test, y_pred_balanced)

print("\nBalanced Logistic Regression Confusion Matrix:")
print(cm_balanced)