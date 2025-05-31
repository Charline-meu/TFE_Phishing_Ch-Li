import pandas as pd
import numpy as np
import torch
import re
from langdetect import detect, LangDetectException
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import average_precision_score, classification_report
import time

# Track start time
start_time = time.time()

# ====== Config ======
FILE_PATHS = ["Data/text_test_set.csv", "Data/text_train_set_20.csv"]
MODEL_DIR = "../final_bert_model"
BATCH_SIZE = 32
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

# ====== Load and Combine Datasets ======
print("Loading and preprocessing datasets...")
df_list = []
for file_path in FILE_PATHS:
    df = pd.read_csv(file_path)[['text', 'label']].dropna()
    df['text'] = df['text'].apply(preprocess_text)
    df = df[df['text'].notna()]
    df_list.append(df)

combined_df = pd.concat(df_list, ignore_index=True)
print(f"Combined dataset size: {len(combined_df)} samples\n")

# ====== Dataset ======
class SpamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ====== Load model and tokenizer ======
print("Loading model and tokenizer...")
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

# ====== DataLoader ======
test_dataset = SpamDataset(combined_df['text'], combined_df['label'], tokenizer, max_len=MAX_LEN)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ====== Evaluation ======
print("Evaluating on combined test set...")
y_true, y_preds, y_probs = [], [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].cpu().numpy()

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[:, 1]
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        y_probs.extend(probs)
        y_preds.extend(preds)
        y_true.extend(labels)

# ====== Metrics ======
auprc = average_precision_score(y_true, y_probs)
report = classification_report(y_true, y_preds)

# Track end time and calculate total runtime
end_time = time.time()
total_time_seconds = end_time - start_time
hours = int(total_time_seconds // 3600)
minutes = int((total_time_seconds % 3600) // 60)
seconds = int(total_time_seconds % 60)
print(f"Total training time: {hours}h {minutes}m {seconds}s")

print(f"\n📈 AUPRC: {auprc:.4f}")
print("\n📋 Classification Report:")
print(report)