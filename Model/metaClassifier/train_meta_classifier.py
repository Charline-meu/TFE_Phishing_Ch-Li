import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

# 1️⃣ Load Preprocessed Meta-Classifier Input
print("📄 Loading combined prediction CSV...")
combined_df = pd.read_csv("meta_classifier_input.csv", index_col="index")

# 2️⃣ Prepare features and labels
labels = combined_df["label"]
proba_modules = combined_df.drop(columns=(["label"]))

# 3️⃣ Handle missing values
print("🧹 Handling missing values (-1 to NaN)...")
proba_modules[proba_modules == -1] = np.nan

# 5️⃣ Train XGBoost Meta-Classifier
print("🚀 Training XGBoost meta-classifier...")
meta_classifier = xgb.XGBClassifier(
    n_estimators=100, random_state=42, use_label_encoder=False,
    eval_metric="logloss", missing=np.nan
)
meta_classifier.fit(proba_modules, labels)
print("✅ Training complete!")

# 7️⃣ Save Model
joblib.dump(meta_classifier, "meta_classifier_model.pkl")
print("💾 Model saved as 'meta_classifier_model.pkl'")