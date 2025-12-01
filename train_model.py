import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import os

# Load CSV: skip meta row if needed
data = pd.read_csv("swine_heat_detection_dataset.csv", skiprows=[2], header=0)

# Show columns loaded
print("Columns loaded:", data.columns.tolist())

# Clean categorical values
for col in ["vulva_swelling", "heat_sign", "behavior_change"]:
    if col in data.columns:
        data[col] = data[col].astype(str).str.replace("\\", "", regex=False).str.strip()
    else:
        raise ValueError(f"Column '{col}' not found! Check your CSV headers.")

# Mapping categorical features
vulva_map = {"None": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
heat_map = {"None": 0, "Mounting Other Sows": 1, "Allowing Mount": 2, "Standing Heat": 3, "Restlessness": 0}
behavior_map = {"Normal": 0, "Aggression": 1, "Loss of Appetite": 2, "Increased Vocalization": 3, "Restlessness": 4}

# Apply mappings
data["vulva_num"] = data["vulva_swelling"].map(vulva_map).fillna(0)
data["heat_num"] = data["heat_sign"].map(heat_map).fillna(0)
data["behavior_num"] = data["behavior_change"].map(behavior_map).fillna(0)

# Combined heat + vulva score
data["heat_vulva_score"] = data["vulva_num"] + data["heat_num"]

# Ensure numeric columns are numeric
numeric_cols = ["activity_level", "temperature_c", "behavior_num", "heat_vulva_score"]
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop rows with missing numeric data
data = data.dropna(subset=numeric_cols)

# Features and target
X = data[numeric_cols]
y = (data["heat_vulva_score"] > 3).astype(int)  # temporary binary target

# Split dataset: 70% train, 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Train Random Forest with 200 trees
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model
os.makedirs("models", exist_ok=True)
with open("models/Heat_Detection_Model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as models/Heat_Detection_Model.pkl")

# Evaluate model on test set
y_pred = model.predict(X_test)
print("Accuracy on test set:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
