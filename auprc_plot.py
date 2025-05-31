import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import numpy as np

# Générer deux jeux de données : un facile (AUPRC élevé), un difficile (AUPRC faible)
X_easy, y_easy = make_classification(
    n_samples=1000, n_classes=2, weights=[0.9, 0.1],
    n_informative=10, class_sep=2.0, random_state=42
)

X_hard, y_hard = make_classification(
    n_samples=1000, n_classes=2, weights=[0.9, 0.1],
    n_informative=2, class_sep=0.5, random_state=42
)

# Entraîner un classificateur simple
clf_easy = LogisticRegression().fit(X_easy, y_easy)
clf_hard = LogisticRegression().fit(X_hard, y_hard)

# Probabilités prédites
y_scores_easy = clf_easy.predict_proba(X_easy)[:, 1]
y_scores_hard = clf_hard.predict_proba(X_hard)[:, 1]

# Calcul des courbes PR
precision_easy, recall_easy, _ = precision_recall_curve(y_easy, y_scores_easy)
precision_hard, recall_hard, _ = precision_recall_curve(y_hard, y_scores_hard)

# Calcul de l'AUPRC
auprc_easy = average_precision_score(y_easy, y_scores_easy)
auprc_hard = average_precision_score(y_hard, y_scores_hard)

# Tracer les courbes
plt.figure(figsize=(10, 5))

# Cas difficile
plt.subplot(1, 2, 1)
plt.plot(recall_hard, precision_hard, color='darkred', lw=2, label=f"AUPRC = {auprc_hard:.2f}")
plt.fill_between(recall_hard, precision_hard, alpha=0.3, color='salmon')
plt.title("Low AUPRC")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid(True)

# Cas facile
plt.subplot(1, 2, 2)
plt.plot(recall_easy, precision_easy, color='darkblue', lw=2, label=f"AUPRC = {auprc_easy:.2f}")
plt.fill_between(recall_easy, precision_easy, alpha=0.3, color='skyblue')
plt.title("High AUPRC")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
