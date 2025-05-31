import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, average_precision_score

# 1️⃣ Load new CSV file with the same structure
csv_path = "test_classifier_input.csv"  # Change this if your new file has a different name
print(f"📥 Loading data from: {csv_path}")
new_data = pd.read_csv(csv_path, index_col="index")

# 2️⃣ Prepare features and labels
labels = new_data["label"]
features = new_data.drop(columns=["label"])

# 3️⃣ Handle missing values
features[features == -1] = np.nan

# 4️⃣ Load the trained model
model_path = "meta_classifier_model.pkl"
print(f"📦 Loading trained model from: {model_path}")
meta_classifier = joblib.load(model_path)

# 5️⃣ Predict
print("🔮 Predicting on new data...")
predictions = meta_classifier.predict(features)

# 6️⃣ Evaluate
print("\n📊 Evaluation Results:")
print(classification_report(labels, predictions))
auprc = average_precision_score(labels, predictions)
print(f"AUPRC: {auprc:.4f}")
