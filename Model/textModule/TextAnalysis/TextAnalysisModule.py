import torch
import time  # Import time module
import re
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from langdetect import detect, LangDetectException
import numpy as np
import os


# Track start time
start_time = time.time()

# Define dataset class
class EmailDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=512):
        self.texts = texts.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }

# Load model and tokenizer
MODEL_DIR = "../final_bert_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

print("Loading model and tokenizer...")
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
print("Loading fine-tuned model...")
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()
print("Model loaded successfully!\n")

# Load new dataset
NEW_DATASET_PATH = "Data/text_test_set.csv"  # Update this path
print(f"Loading new email dataset from {NEW_DATASET_PATH}")
new_df = pd.read_csv(NEW_DATASET_PATH)
new_df = new_df[new_df['label'] == 1].sample(n=12, random_state=42)
print(f"Dataset loaded! {len(new_df['text'])} emails found.\n")

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

# Save preprocessed emails and labels into a CSV file
# OUTPUT_CSV = "../ExplainableAI/output_emails.csv"
# output_df = new_df[['text', 'label']]  # Use original text and original label
# output_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
# print(f"Preprocessed emails saved to CSV: {OUTPUT_CSV}\n")

# Apply text preprocessing
print("Preprocessing email content...")
# Save original text before preprocessing
new_df['original_text'] = new_df['text']
# Apply text preprocessing
new_df['text'] = new_df['original_text'].apply(preprocess_text)
new_df = new_df[new_df['text'].notna()]  # Remove NaN entries
new_texts = new_df['text']
print(f"Preprocessed dataset size: {len(new_df['text'])} emails\n")

# Create DataLoader
print("Tokenizing and creating DataLoader...")
new_dataset = EmailDataset(new_texts, tokenizer)
new_loader = DataLoader(new_dataset, batch_size=32, shuffle=False)
print("DataLoader ready!\n")

# Predict probabilities
print("Starting inference...")
pred_probs = []
with torch.no_grad():
    for i, batch in enumerate(new_loader):
        print(f"Processing batch {i+1}/{len(new_loader)}...")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        probs = F.softmax(outputs.logits, dim=1)[:, 1]  # Get positive class probabilities
        pred_probs.extend(probs.cpu().numpy())

print("Inference completed!\n")

# Save predictions
print("Saving predictions to CSV...")
new_df['text_proba'] = pred_probs
OUTPUT_PATH = "../../test_with_predictions.csv"
# Rename preprocessed text column for clarity
new_df = new_df.rename(columns={'text': 'preprocessed_text'})

# Reorder and save desired columns
final_df = new_df[['original_text', 'preprocessed_text', 'label', 'text_proba']]
final_df.to_csv(OUTPUT_PATH, index=True)

print(f"Predictions saved to {OUTPUT_PATH}")

# Print AUPRC
auprc = average_precision_score(new_df["label"], pred_probs)
print(f"AUPRC: {auprc:.4f}")

# Track end time and calculate total runtime
end_time = time.time()
total_time_seconds = end_time - start_time
hours = int(total_time_seconds // 3600)
minutes = int((total_time_seconds % 3600) // 60)
seconds = int(total_time_seconds % 60)
print(f"Total runtime: {hours} hours, {minutes} minutes, {seconds} seconds") 