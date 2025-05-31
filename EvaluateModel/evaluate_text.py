import torch
import re
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from langdetect import detect, LangDetectException
import numpy as np

# --------------------- CONFIG -----------------------
OUTPUT_CSV = "predictions_emails.csv"
TEXT_MODEL_NAME = "bert-base-uncased"
TEXT_MODEL_SAVE_PATH = "textModule/BERT_fine-tuned/best_bert_spam_classifier_auprc.pth"
TEXT_FOLDER_PATH = Path("text_outputs")  # 📁 Folder with all text files
# ----------------------------------------------------

# Preprocessing function
def preprocess_text(text):
    if text == "-1":
        return np.nan  # Return NaN for invalid "-1" entries

    try:
        if detect(text) != 'en':
            return np.nan  # Return NaN for non-English entries
    except LangDetectException:
        return np.nan  # Return NaN when detection fails

    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Remove URLs
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = re.sub(r"\S+@\S+\.\S+", "", text)  # Remove email addresses
    #text = re.sub(r"[^A-Za-z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Normalize spaces
    return text

# Load model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(TEXT_MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(TEXT_MODEL_NAME, num_labels=2)
model.load_state_dict(torch.load(TEXT_MODEL_SAVE_PATH, map_location=device))
model.to(device)
model.eval()

# Load CSV
df = pd.read_csv(OUTPUT_CSV)

# Loop over each file like text_output_0_....txt
for txt_file in TEXT_FOLDER_PATH.glob("text_output_*.txt"):
    try:
        # Extract ID from filename (e.g., text_output_3_xxx.txt → id = 3)
        id_str = txt_file.stem.split("_")[2]  # "3"
        idx = int(id_str)

        # Read and process text
        with open(txt_file, "r", encoding="utf-8") as f:
            input_text = f.read()
        preprocessed = preprocess_text(input_text)
        inputs = tokenizer(preprocessed, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            text_proba = probs[0, 1].item()

        # Update CSV
        if idx in df["id"].values:
            df.loc[df["id"] == idx, "text_proba"] = text_proba
            print(f"✅ Updated id {idx} in CSV with text_proba = {text_proba:.4f}")
        else:
            print(f"⚠️ id {idx} not found in CSV")

    except Exception as e:
        print(f"⚠️ Error processing {txt_file.name}: {e}")

# Save updated CSV
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n📁 All text_proba values updated in {OUTPUT_CSV}")

# Delete all files in the text_outputs folder
print("\n🧹 Cleaning up text_outputs folder...")
for txt_file in TEXT_FOLDER_PATH.glob("text_output_*.txt"):
    try:
        txt_file.unlink()
        print(f"🗑️ Deleted: {txt_file.name}")
    except Exception as e:
        print(f"⚠️ Could not delete {txt_file.name}: {e}")

print("✅ Cleanup complete.")