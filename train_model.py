import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset
print("Loading data...")
df = pd.read_csv("HR-Employee-Attrition.csv")
print(df.head())

# Encode categorical features
print("Encoding categorical columns...")
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and target
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("Training the model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

# Save model, encoders, and feature names
print("Saving the model...")
joblib.dump(model, "attrition_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

with open("model_features.pkl", "wb") as f:
    joblib.dump(X_train.columns.tolist(), f)

print("Model and features saved successfully.")






