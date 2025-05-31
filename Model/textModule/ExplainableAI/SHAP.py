import pandas as pd
import numpy as np
import torch
import re
from langdetect import detect, LangDetectException
from transformers import BertTokenizer, BertForSequenceClassification
import time
import shap
import matplotlib.pyplot as plt
import os


# Track start time
start_time = time.time()

# ====== Config ======
#FILE_PATH = "output_emails.csv"  # Using just one CSV file
FILE_PATH = "Data/text_test_set.csv"  # Using just one CSV file
MODEL_DIR = "../final_bert_model"
MAX_LEN = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ====== Preprocessing ======
def preprocess_text(text):
    if text == "-1":
        return np.nan
    try:
        if detect(text) != 'en':
            return np.nan
    except LangDetectException:
        return np.nan
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\S+@\S+\.\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ====== Load and Preprocess Dataset ======
print("Loading and preprocessing dataset...")
df = pd.read_csv(FILE_PATH)[['text', 'label']].dropna()
df['text'] = df['text'].apply(preprocess_text)
df = df[df['text'].notna()]
print(f"Dataset size: {len(df)} samples\n")

# ====== Load model and tokenizer ======
print("Loading model and tokenizer...")
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

# ====== SHAP Analysis ======
print("\n===== SHAP Explainability Analysis =====")

# Function to get model predictions for SHAP
def predict_proba(texts):
    """Tokenizes input texts and returns model probability predictions."""
    if isinstance(texts, str):  # Convert a single string input to a list
        texts = [texts]
    
    # Tokenize input text (convert to BERT-compatible format)
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    
    # Move tensors to the same device as the model
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs).logits  # Get raw logits
        return torch.softmax(outputs, dim=1).cpu().numpy()  # Convert logits to probabilities

# Function to extract top influential features from SHAP values
def get_top_directional_features(shap_values, top_n=20):
    """Extract top tokens that push towards phishing (positive) and legitimate (negative)."""
    values = shap_values.values[0, :, 1]  # For the second class (likely phishing, index 1)
    tokens = shap_values.data[0]
    
    # Create tuples of (token, value)
    feature_importance = [(token, value) for token, value in zip(tokens, values)]
    
    # Sort by absolute importance for visualization
    abs_sorted = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)[:top_n]
    
    # Also get top positive (towards phishing) and negative (towards legitimate) features
    positive_sorted = sorted(feature_importance, key=lambda x: x[1], reverse=True)[:top_n]
    negative_sorted = sorted(feature_importance, key=lambda x: x[1])[:top_n]
    
    return {
        'overall': abs_sorted,
        'towards_phishing': positive_sorted,
        'towards_legitimate': negative_sorted
    }

# Get samples labeled as phishing (label=1)
selected_phishing = df[df['label'] == 1]['text'].sample(n=100, random_state=42)
#selected_phishing = df["text"]

print(f"Selected {len(selected_phishing)} samples for analysis")

# Create a Text masker for SHAP
print("Creating SHAP explainer...")
masker = shap.maskers.Text(tokenizer)

# Create the SHAP Explainer
explainer = shap.Explainer(
    lambda x: predict_proba([str(t) for t in x]), 
    masker
)

# Output folder for SHAP reports
OUTPUT_DIR = "shap_reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Process each phishing sample
for i, text in enumerate(selected_phishing):
    sample_label = f"Sample {i+1}"
    print(f"\n{'='*50}")
    print(f"Analyzing {sample_label}...")
    print(f"Sample text: {text[:150]}{'...' if len(text) > 150 else ''}")
    
    # Compute SHAP values
    shap_values = explainer([text])
    
    # Predict
    pred = predict_proba(text)[0]
    pred_class = 'phishing' if pred[1] > pred[0] else 'legitimate'
    pred_confidence = max(pred[0], pred[1])

    # Get top features
    top_features = get_top_directional_features(shap_values)

    # Prepare output string
    output_lines = [
        f"{sample_label}",
        f"{'=' * len(sample_label)}\n",
        f"Sample text preview: {text[:150]}{'...' if len(text) > 150 else ''}\n",
        f"Model prediction: {pred}",
        f"Predicted class: {pred_class.upper()} with {pred_confidence:.4f} confidence\n",
        "🔍 Top 20 influential features:\n",
        "Top features by absolute importance:",
    ]

    for idx, (token, value) in enumerate(top_features['overall'], 1):
        direction = "→ PHISHING" if value > 0 else "→ LEGITIMATE"
        output_lines.append(f"{idx:2d}. {token:20s} | Impact: {value:.6f} {direction}")

    output_lines.append("\nTop features pushing towards PHISHING classification:")
    for idx, (token, value) in enumerate(top_features['towards_phishing'], 1):
        if value > 0:
            output_lines.append(f"{idx:2d}. {token:20s} | Impact: {value:.6f}")

    output_lines.append("\nTop features pushing towards LEGITIMATE classification:")
    for idx, (token, value) in enumerate(top_features['towards_legitimate'], 1):
        if value < 0:
            output_lines.append(f"{idx:2d}. {token:20s} | Impact: {value:.6f}")

    # Write to file
    filename = os.path.join(OUTPUT_DIR, f"shap_report_sample_{i+1:02d}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"✅ Saved SHAP report to {filename}")

# Track end time and calculate total runtime
end_time = time.time()
total_time_seconds = end_time - start_time
hours = int(total_time_seconds // 3600)
minutes = int((total_time_seconds % 3600) // 60)
seconds = int(total_time_seconds % 60)

print(f"\nTotal runtime: {hours}h {minutes}m {seconds}s")
print("\n✅ SHAP analysis complete! Check the saved visualization files for token-level explainability.")