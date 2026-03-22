import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv("dump_data.csv")

# -------------------------
# Feature Engineering
# -------------------------
df["efficiency_ratio"] = df["max_angle"] / df["total_dump_time"]
df["vibration_effectiveness"] = df["vibration_peak"] / df["hold_time"]
df["dump_aggressiveness"] = df["vibration_rms"] * df["max_angle"]

# -------------------------
# Features & Targets
# -------------------------
feature_cols = [
    "max_angle",
    "hold_time",
    "total_dump_time",
    "vibration_rms",
    "vibration_peak",
    "initial_load",
    "efficiency_ratio",
    "vibration_effectiveness",
    "dump_aggressiveness"
]

X = df[feature_cols]

y_reg = df["leftover_percent"]
y_clf = df["severity"]

# -------------------------
# Encode labels
# -------------------------
le = LabelEncoder()
y_clf_encoded = le.fit_transform(y_clf)

# -------------------------
# Train-test split
# -------------------------
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf_encoded, test_size=0.2, random_state=42
)

# -------------------------
# Regression Model
# -------------------------
reg_model = RandomForestRegressor(n_estimators=150, random_state=42)
reg_model.fit(X_train, y_reg_train)

y_pred_reg = reg_model.predict(X_test)
mae = mean_absolute_error(y_reg_test, y_pred_reg)

print("✅ Regression MAE:", round(mae, 2))

# -------------------------
# Classification Model
# -------------------------
clf_model = RandomForestClassifier(n_estimators=150, random_state=42)
clf_model.fit(X_train, y_clf_train)

y_pred_clf = clf_model.predict(X_test)
accuracy = accuracy_score(y_clf_test, y_pred_clf)

print("✅ Classification Accuracy:", round(accuracy, 2))

# -------------------------
# Sample Prediction
# -------------------------
sample = pd.DataFrame([{
    "max_angle": 50,
    "hold_time": 4,
    "total_dump_time": 8,
    "vibration_rms": 1.5,
    "vibration_peak": 2.0,
    "initial_load": 1000,
    "efficiency_ratio": 50/8,
    "vibration_effectiveness": 2.0/4,
    "dump_aggressiveness": 1.5*50
}])

leftover = reg_model.predict(sample)[0]
severity = le.inverse_transform(clf_model.predict(sample))[0]

print("\n🔍 Test Prediction:")
print("Predicted leftover:", round(leftover, 2))
print("Severity:", severity)

# -------------------------
# Save Models
# -------------------------
joblib.dump(reg_model, "reg_model.pkl")
joblib.dump(clf_model, "clf_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("💾 Models saved successfully!")