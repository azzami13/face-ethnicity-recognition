import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# Load embeddings
with open("embeddings.json", "r") as f:
    embeddings = json.load(f)

labels = []
vectors = []

for fname, emb in embeddings.items():
    vectors.append(emb)
    name = fname.split("_")[1]  # contoh: datar_andi_sunda_xxx -> ambil 'andi'
    labels.append(name)

vectors = np.array(vectors)
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
reduced = tsne.fit_transform(vectors)

# Visualisasi
plt.figure(figsize=(10, 7))
for i, label in enumerate(set(labels)):
    idxs = [j for j, l in enumerate(labels) if l == label]
    plt.scatter(reduced[idxs, 0], reduced[idxs, 1], label=label)
plt.legend()
plt.title("t-SNE Visualization of Face Embeddings")
plt.show()
