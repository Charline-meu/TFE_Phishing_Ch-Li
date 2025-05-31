import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import shap
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("structureModule/extraction_features_spam_assassin.csv")

# Separate features and target
X = df.drop(columns=['label'])
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Initialize the SHAP explainer
explainer = shap.Explainer(rf_model, X_train)

# Calculate SHAP values for the test set
shap_values = explainer(X_test)

# For binary classification, shap_values.values has shape (n_samples, n_features, 2)
# We select the SHAP values for class 1 (positive class)
shap_values_class1 = shap_values[..., 1]

# Create a beeswarm plot for class 1
shap.plots.beeswarm(shap_values_class1, max_display=30)

shap.plots.bar(shap_values_class1.mean(0), max_display=30)
