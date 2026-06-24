import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# Loading the Datasets

df = pd.read_csv('fleet_health.csv')

# Defining the input (x) and target (y) variables

# 1.  These are our input features

X = df[['Age',
        "Total_KM_Driven",
         "Fuel_Efficiency",
         "Battery_Health",
         "Service_Delay",
         "Engine_Temp",
         "Repair_Count" ]]

# 2. Labels these are our target variables - its what we want to predict

y = df['Status']

print("Features Shape:", X.shape),
print("Target Shape:", y.shape)
print("Unique Values in Target Variable:", y.unique())



# Encoding the target column - changing the text column to the number

le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("Encoding the Status column:")
for i, label in enumerate(le.classes_):
    print (f"{label} : {i}")


# Splitting the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
print("Training Set Shape:", X_train.shape),
print("Testing Set Shape:", X_test.shape)

# Training the model

model = RandomForestClassifier(
    n_estimators = 100,
    random_state = 42,\
    n_jobs=-1
)

print("Training the model.......")
model.fit(X_train, y_train)
print("Model Trained Successfully")

# Make prediction on test set

y_pred = model.predict(X_test)

# Model Evaluation and finding accuracy

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Making a Detailed Report

print(classification_report(y_test, y_pred, target_names=le.classes_))


# Which feature does the model rely on the most

feature_importance = pd.DataFrame({"Feature": X.columns,
"Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)


print("\nFeature Importance (which columns matter most):")
print(feature_importance)


# Saving model to file

joblib.dump(model, 'model.pkl')
print("Model saved successfully!")


# Saving the label encoder to file

joblib.dump(le, 'label_encoder.pkl')
print("Label encoder saved successfully!")

print('All done!')