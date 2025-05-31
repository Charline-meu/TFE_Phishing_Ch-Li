import numpy as np
import pandas as pd
import torch
import re
from langdetect import detect, LangDetectException
from transformers import BertTokenizer, BertForSequenceClassification
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer

# Load your model and tokenizer
MODEL_DIR = "../final_bert_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

print(f"Using device: {device}")

# Text preprocessing function - copied from your original code
def preprocess_text(text):
    if text == "-1":
        return np.nan
    try:
        if detect(text) != 'en':
            return np.nan
    except LangDetectException:
        return np.nan
    
    text = re.sub(r"http\S+|www\S+|https\S+", "", text) # Clean URLs
    text = re.sub(r"\d+", "", text) # Clean numbers
    text = re.sub(r"\S+@\S+\.\S+", "", text) # Clean email addresses
    text = re.sub(r"\s+", " ", text).strip() # Clean extra whitespace
    
    return text

# Load a sample email from your dataset
df_sample = pd.read_csv("output_emails.csv")
# Apply preprocessing
df_sample['processed_text'] = df_sample['original_text'].apply(preprocess_text)
df_sample['processed_text'] = df_sample['processed_text'].iloc[8] 
# Remove rows with NaN values after preprocessing
df_sample = df_sample.dropna(subset=['processed_text'])

if len(df_sample) == 0:
    print("Error: All samples were filtered out during preprocessing. Try a larger sample size.")
    exit()

texts = df_sample['processed_text'].tolist()
labels = df_sample['label'].tolist()

# Create a prediction function that LIME can use
def predict_proba(texts):
    # Convert to list if it's a single string
    if isinstance(texts, str):
        texts = [texts]
    
    # Process inputs in smaller batches to avoid memory issues
    batch_size = 8
    all_probs = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
            
        all_probs.append(probs)
    
    # Combine results from all batches
    all_probs = np.vstack(all_probs)
    return all_probs

import os

# ====== Create output directory ======
output_dir = "lime_outputs"
os.makedirs(output_dir, exist_ok=True)

# ====== LIME Explainer Setup ======
class_names = ['Legitimate', 'Phishing']
explainer = LimeTextExplainer(class_names=class_names)
num_features = 25  # Number of features to include in the explanation

# ====== Loop over all samples ======
for idx, (email_text, true_label) in enumerate(zip(texts, labels)):
    if not isinstance(email_text, str):
        email_text = str(email_text)

    print(f"🔍 Explaining email {idx+1}/{len(texts)}...")

    # Predict
    proba = predict_proba([email_text])[0]
    pred_class = np.argmax(proba)

    # Generate explanation
    exp = explainer.explain_instance(
        email_text, 
        predict_proba, 
        num_features=num_features,
        num_samples=500
    )

    # Save explanation to a text file
    txt_filename = os.path.join(output_dir, f"lime_email_{idx+1}.txt")
    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write(f"=== LIME Explanation for Email #{idx+1} ===\n")
        f.write(f"True Label: {'Phishing' if true_label == 1 else 'Legitimate'}\n")
        f.write(f"Model Prediction: {'Phishing' if pred_class == 1 else 'Legitimate'}\n")
        f.write(f"Prediction Probability: {proba[pred_class]:.4f}\n\n")
        f.write("Top Features Influencing the Prediction:\n")
        for feature, score in exp.as_list():
            direction = "Phishing" if score > 0 else "Legitimate"
            f.write(f"{feature}: {score:.4f} (pushes toward {direction})\n")

    # Recreate LIME plot with reversed colors: red = phishing, green = legitimate
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # Get features and weights
    feature_list = exp.as_list()
    features = [f[0] for f in feature_list]
    weights = [f[1] for f in feature_list]

    # Define custom colors
    colors = ['red' if w > 0 else 'green' for w in weights]

    # Plot
    plt.figure(figsize=(8, 6))
    y_pos = np.arange(len(features))
    plt.barh(y_pos, weights, color=colors)
    plt.yticks(y_pos, features)
    plt.xlabel('Feature Contribution')
    plt.title('Local explanation for class Phishing')
    plt.axvline(0, color='black', linewidth=0.5)

    # Invert y-axis to match LIME's default orientation
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"lime_plot_{idx+1}.png"), dpi=300)
    plt.close()

print(f"\n✅ All explanations saved to '{output_dir}/'")

# # Generate explanation
# num_features = 25  # Number of features to include in the explanation
# exp = explainer.explain_instance(
#     email_text, 
#     predict_proba, 
#     num_features=num_features,
#     num_samples=500
# )

# # Plot the explanation
# plt.figure(figsize=(10, 6))
# exp.as_pyplot_figure()
# plt.tight_layout()
# plt.savefig('lime_explanation.png', dpi=300)
# plt.show()

# # Print the explanation
# print("\nTop features influencing the prediction:")
# for feature, score in exp.as_list():
#     direction = "Phishing" if score > 0 else "Legitimate"
#     print(f"{feature}: {score:.4f} (pushes toward {direction})")

# # Alternative bar chart visualization
# def plot_lime_features(exp, class_idx=1):  # class_idx=1 for "Phishing" class
#     # Get the explanation for the specified class
#     exp_list = exp.as_list(label=class_idx)
    
#     # Sort by absolute importance
#     exp_list = sorted(exp_list, key=lambda x: abs(x[1]), reverse=True)
    
#     # Extract features and scores
#     features = [x[0] for x in exp_list]
#     scores = [x[1] for x in exp_list]
    
#     # Set colors based on scores
#     colors = ['blue' if s > 0 else 'red' for s in scores]
    
#     # Create plot
#     plt.figure(figsize=(10, 6))
#     y_pos = np.arange(len(features))
#     plt.barh(y_pos, scores, color=colors)
#     plt.yticks(y_pos, features)
#     plt.xlabel('Feature Importance')
#     plt.title('LIME Feature Importance\nBlue: Pushes toward Phishing, Red: Pushes toward Legitimate')
#     plt.tight_layout()
#     plt.savefig('lime_bar_chart.png', dpi=300)
#     plt.show()

# # Generate custom bar chart
# plot_lime_features(exp)

