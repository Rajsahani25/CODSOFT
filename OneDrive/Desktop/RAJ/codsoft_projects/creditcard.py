import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_csv(r"C:\Users\Acer25\OneDrive\Desktop\RAJ\codsoft_projects\CREDIT CARD FRAUD DETECTION-5\creditcard.csv\creditcard.csv")

# Feature scaling for 'Amount' and drop 'Time'
df['Amount'] = StandardScaler().fit_transform(df[['Amount']])
df = df.drop(['Time'], axis=1)

# Define features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance using SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Train models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# Evaluate each model
for name, model in models.items():
    print(f"\n{name}")
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    roc_auc = roc_auc_score(y_test, y_pred)
    print(f"ROC AUC Score: {roc_auc:.4f}")
