import pandas as pd
from joblib import dump
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("datasets/dataset.csv")

y = df["Disease"]

X = pd.read_csv("datasets/processed.csv")

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)
gb.fit(X_train, y_train)
rf.fit(X_train, y_train)

dump(le, "le.joblib")
dump(dt, "models/dt.joblib")
dump(gb, "models/gb.joblib")
dump(rf, "models/rf.joblib")

y_pred = dt.predict(X_test)
y_pred1 = gb.predict(X_test)
y_pred2 = rf.predict(X_test)

print(f"DT Predictions: {le.inverse_transform(y_pred)}")
print(f"RF Predictions: {le.inverse_transform(y_pred2)}")
print(f"GB Predictions: {le.inverse_transform(y_pred1)}")

accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Classifier Accuracy: {accuracy:.2f}")

print("\nDecision Tree Classifier Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

accuracy = accuracy_score(y_test, y_pred2)
print(f"Random Forest Accuracy: {accuracy:.2f}")

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred2, target_names=le.classes_))

accuracy = accuracy_score(y_test, y_pred1)
print(f"Gradient Boost Accuracy: {accuracy:.2f}")

print("\nGradient Boost Classification Report:")
print(classification_report(y_test, y_pred1, target_names=le.classes_))
