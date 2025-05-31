import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, average_precision_score

# --------------------- CONFIG -----------------------

OUTPUT_CSV = "predictions_emails.csv"
MODEL_PATH = "metaClassifier/meta_classifier_model.pkl"

# ----------------------------------------------------

# 1️⃣ Load new CSV file with the same structure
print(f"📥 Loading data from: {OUTPUT_CSV}")
new_data = pd.read_csv(OUTPUT_CSV)

# 2️⃣ Prepare features and labels
features = new_data.drop(columns=["filename", "id"])

# 3️⃣ Handle missing values
features[features == -1] = np.nan

# 4️⃣ Load the trained model
print(f"📦 Loading trained model from: {MODEL_PATH}")
meta_classifier = joblib.load(MODEL_PATH)

# 5️⃣ Predict
print("🔮 Predicting on new data...")
predictions = meta_classifier.predict(features)
print("🔮 Predictions: ", predictions)

# 6️⃣ Append predictions to DataFrame
new_data["meta_prediction"] = predictions

# 7️⃣ Save updated DataFrame to CSV
new_data.to_csv(OUTPUT_CSV, index=False)  # Overwrite original
# Or save to a new file:
# new_data.to_csv(FINAL_OUTPUT_CSV, index=False)

print(f"✅ Predictions appended to: {OUTPUT_CSV}")