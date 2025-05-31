import torch
import torch.nn.functional as F
import re # Python’s Regular Expression Package
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from textattack.models.wrappers import ModelWrapper
from textattack.attack_recipes import PWWSRen2019
from textattack.attack_recipes import DeepWordBugGao2018
from textattack.attack_recipes import TextFoolerJin2019
from textattack.datasets import Dataset
from textattack.attack_args import AttackArgs
from textattack import Attacker
from langdetect import detect, LangDetectException
import time  # Import time module
import numpy as np

# Track start time
start_time = time.time()

# ---------- Custom TextAttack Model Wrapper ---------- #
class CustomBERTWrapper(ModelWrapper):
    def __init__(self, model, tokenizer, device):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def preprocess(self, text):
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

    def __call__(self, text_list):
        inputs = self.tokenizer(
            [self.preprocess(text) for text in text_list],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
        return probs.cpu()

# ---------- Load your custom fine-tuned BERT model ---------- #
MODEL_DIR = "final_bert_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

print("Loading model and tokenizer...")
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()
print("Model loaded successfully!\n")

# Wrap it
model_wrapper = CustomBERTWrapper(model, tokenizer, device)

# ---------- Define the Attack ---------- #
#attack = DeepWordBugGao2018.build(model_wrapper)
#attack = PWWSRen2019.build(model_wrapper)
attack = TextFoolerJin2019.build(model_wrapper)

# ---------- Prepare Custom Dataset ---------- #
# You can load from CSV or hardcode samples here
df = pd.read_csv("../test_with_predictions.csv")

texts = df["preprocessed_text"].tolist()
labels = df["label"].tolist()

print(f"Filtered dataset: {len(texts)} valid texts loaded.")

# Create (text, label) tuples
dataset = Dataset(list(zip(texts, labels)))

# ---------- Run the Attack ---------- #
attack_args = AttackArgs(
    num_examples=10,
    log_to_csv="attack_results_custom_model.csv",
    disable_stdout=False,
    random_seed=42
)

attacker = Attacker(attack, dataset, attack_args)
attacker.attack_dataset()

# Track end time and calculate total runtime
end_time = time.time()
total_time_seconds = end_time - start_time
hours = int(total_time_seconds // 3600)
minutes = int((total_time_seconds % 3600) // 60)
seconds = int(total_time_seconds % 60)
print(f"Total runtime: {hours} hours, {minutes} minutes, {seconds} seconds") 