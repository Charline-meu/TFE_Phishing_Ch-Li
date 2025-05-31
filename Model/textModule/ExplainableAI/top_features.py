import re
import os
from collections import defaultdict
import matplotlib.pyplot as plt

num_features = 20

# ====== Config ======
INPUT_DIR = "shap_reports"
OUTPUT_TXT_PATH = "top_features_plot/top" + str(num_features) + "_features.txt"

# Output folder for SHAP reports
OUTPUT_DIR = "top_features_plot"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====== Functions ======

def extract_all_features_by_direction(section_name, lines):
    """Extracts (token, impact) pairs from a specific section."""
    all_features = []
    current_idx = 0
    while current_idx < len(lines):
        if section_name in lines[current_idx]:
            current_idx += 1
            while current_idx < len(lines):
                line = lines[current_idx].strip()
                if not line or line.startswith("Top features pushing"):
                    break
                match = re.match(r"\s*\d+\.\s+(.*?)\s+\|\s+Impact:\s+([-+]?[0-9]*\.?[0-9]+)", line)
                if match:
                    token, impact = match.groups()
                    token = token.strip() or "[EMPTY]"
                    all_features.append((token, float(impact)))
                current_idx += 1
        current_idx += 1
    return all_features

# ====== Aggregate Across Files ======
phishing_impact = defaultdict(float)
legitimate_impact = defaultdict(float)

for filename in os.listdir(INPUT_DIR):
    if filename.endswith(".txt"):
        filepath = os.path.join(INPUT_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        phishing_features = extract_all_features_by_direction(
            "Top features pushing towards PHISHING classification:", lines)
        for token, impact in phishing_features:
            phishing_impact[token] += abs(impact)

        legitimate_features = extract_all_features_by_direction(
            "Top features pushing towards LEGITIMATE classification:", lines)
        for token, impact in legitimate_features:
            legitimate_impact[token] += -abs(impact)
    
# ====== Combine and Sort ======
all_tokens = set(phishing_impact.keys()) | set(legitimate_impact.keys())

combined = []

for token in all_tokens:
    phish = phishing_impact[token]
    legit = legitimate_impact[token]
    total = phish + legit
    direction = "PHISHING" if total > 0 else "LEGITIMATE"
    combined.append((token, total, direction))

top_features = sorted(combined, key=lambda x: abs(x[1]), reverse=True)[:num_features]

# ====== Save to TXT ======
with open(OUTPUT_TXT_PATH, "w", encoding="utf-8") as f:
    f.write("Top "+ str(num_features) +" Most Influential Features Across All Emails:\n")
    f.write("Rank | Feature | Total Impact | Direction\n")
    f.write("-" * 50 + "\n")
    for i, (token, total_impact, direction) in enumerate(top_features, 1):
        f.write(f"{i:>2}. {token} | {total_impact:.6f} | {direction}\n")

print("✅ Top "+ str(num_features) +" influential features with direction saved to:", OUTPUT_TXT_PATH)

# ====== Create Bar Plot (PHISHING = green, LEGITIMATE = red) ======
plot_features = []
for token, total_impact, direction in top_features:
    signed_impact = total_impact #if direction == "PHISHING" else -total_impact
    plot_features.append((token, signed_impact))

# Tri par valeur absolue
plot_features = sorted(plot_features, key=lambda x: abs(x[1]), reverse=True)

tokens, signed_impacts = zip(*plot_features)
colors = ['red' if val > 0 else 'green' for val in signed_impacts]  # PHISHING = red, LEGITIMATE = green

plt.figure(figsize=(12, 6))
bars = plt.barh(tokens[::-1], signed_impacts[::-1], color=colors[::-1])
plt.axvline(0, color='black', linewidth=0.8)
plt.xlabel("Signed SHAP Impact")
plt.title("Top "+ str(num_features) +" Influential Features (PHISHING = Red, LEGITIMATE = Green)")
plt.tight_layout()
plt.savefig("top_features_plot/top"+ str(num_features) +"_features_barplot.png")
plt.close()

print("✅ Colored barplot saved as: top"+ str(num_features) +"_features_barplot.png")