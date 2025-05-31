import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# 1️⃣ Load the saved meta-classifier model
model_path = "meta_classifier_model.pkl"
print(f"📦 Loading model from: {model_path}")
meta_classifier = joblib.load(model_path)

# 2️⃣ Load the input data (same format as training)
csv_path = "test_classifier_input.csv"  # Use your input CSV here
print(f"📄 Loading input data from: {csv_path}")
df = pd.read_csv(csv_path, index_col="index")

# 3️⃣ Prepare data
features = df.drop(columns=["label"])
features[features == -1] = np.nan  # Handle missing values

# 4️⃣ Run SHAP
print("🔍 Running SHAP to explain predictions...")
explainer = shap.Explainer(meta_classifier, features)
shap_values = explainer(features)

# Bar plot (global feature importance)
plt.figure()
shap.summary_plot(shap_values, features, plot_type="bar", show=False)
plt.savefig("shap_summary_bar.png", bbox_inches="tight")
plt.close()
print("✅ Saved bar plot as shap_summary_bar.png")

# Beeswarm plot (distribution of impacts)
plt.figure()
shap.summary_plot(shap_values, features, show=False)
plt.savefig("shap_summary_beeswarm.png", bbox_inches="tight")
plt.close()
print("✅ Saved beeswarm plot as shap_summary_beeswarm.png")