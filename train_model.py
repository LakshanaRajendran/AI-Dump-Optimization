import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
from utils import efficiency_score, estimate_energy
import joblib
# Load dataset
df = pd.read_csv("dump_data.csv")

df["efficiency_ratio"] = df["max_angle"] / df["total_dump_time"]
df["vibration_effectiveness"] = df["vibration_peak"] / df["hold_time"]
df["dump_aggressiveness"] = df["vibration_rms"] * df["max_angle"]
# Features
X = df[[
    "max_angle",
    "hold_time",
    "total_dump_time",
    "vibration_rms",
    "vibration_peak",
    "initial_load",
    "efficiency_ratio",
    "vibration_effectiveness",
    "dump_aggressiveness"
]]

# Targets
y_reg = df["leftover_percent"]
y_clf = df["severity"]

# Encode classification labels
le = LabelEncoder()
y_clf_encoded = le.fit_transform(y_clf)

# Split data
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf_encoded, test_size=0.2, random_state=42
)

# ------------------------
# Regression Model
# ------------------------
reg_model = RandomForestRegressor(n_estimators=100)
reg_model.fit(X_train, y_reg_train)

y_pred_reg = reg_model.predict(X_test)
mae = mean_absolute_error(y_reg_test, y_pred_reg)

print("Regression MAE:", mae)

# ------------------------
# Classification Model
# ------------------------
clf_model = RandomForestClassifier(n_estimators=100)
clf_model.fit(X_train, y_clf_train)

y_pred_clf = clf_model.predict(X_test)
accuracy = accuracy_score(y_clf_test, y_pred_clf)

print("Classification Accuracy:", accuracy)

sample = [[50, 4, 8, 1.5, 2.0, 1000, 50/8, 2.0/4, 1.5*50]]

leftover = reg_model.predict(sample)[0]
severity = le.inverse_transform(clf_model.predict(sample))[0]

print("\nTest Prediction:")
print("Predicted leftover:", leftover)
print("Severity:", severity)

eff = efficiency_score(leftover)
energy = estimate_energy(50, 4, 1.5)

print("Efficiency Score:", eff)
print("Estimated Energy:", energy)

# adaptive learning (append new data)

new_row = df.iloc[0:1]  # dummy example

df = pd.concat([df, new_row], ignore_index=True)

print("Dataset updated with new data!")
# Save models
joblib.dump(reg_model, "reg_model.pkl")
joblib.dump(clf_model, "clf_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("Models saved successfully!")
