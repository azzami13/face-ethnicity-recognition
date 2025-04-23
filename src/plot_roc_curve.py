import json
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Load embeddings
with open("embeddings.json", "r") as f:
    embeddings = json.load(f)

names = list(embeddings.keys())
vectors = np.array([embeddings[n] for n in names])

# Ground truth: label dari nama orangnya (contoh: 'andi')
labels = [n.split("_")[1] for n in names]

# Generate true labels and scores
y_true = []
y_scores = []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        same = int(labels[i] == labels[j])
        sim = cosine_similarity([vectors[i]], [vectors[j]])[0][0]
        y_true.append(same)
        y_scores.append(sim)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Face Verification")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
