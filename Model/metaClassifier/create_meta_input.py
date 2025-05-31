import pandas as pd
import numpy as np

print("🔄 Loading module predictions...")

# Load individual module predictions
#structure_preds = pd.read_csv("../enron_spamassassin_nazario_nigerian_best_compo/extracted_features/features_proba_train_set_20.csv")
structure_preds = pd.read_csv("../enron_spamassassin_nazario_nigerian_best_compo/extracted_features/features_proba_test_set.csv")

#text_preds = pd.read_csv("../textModule/MetaTrainer/meta_with_predictions.csv")
text_preds = pd.read_csv("../textModule/MetaTrainer/test_with_predictions.csv")

#url_preds = pd.read_csv("../enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/nlp_proba_train_set_20.csv")
url_preds = pd.read_csv("../enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/nlp_proba_test_set.csv")

# Set 'index' as index for alignment
structure_preds.set_index('index', inplace=True)
text_preds.set_index('index', inplace=True)
url_preds.set_index('index', inplace=True)

# Merge based on index
combined_df = structure_preds[['structure_proba']].join(
    text_preds[['text_proba']], how='inner'
).join(
    url_preds[['url_proba']], how='inner'
)

# Add label from any of the sources (same across them)
combined_df['label'] = structure_preds['label']

# Save to CSV
#output_path = "meta_classifier_input.csv"
output_path = "test_classifier_input.csv"
combined_df.to_csv(output_path)
print(f"✅ Combined predictions saved to: {output_path}")