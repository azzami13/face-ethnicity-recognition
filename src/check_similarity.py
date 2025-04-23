import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

with open("embeddings.json", "r") as f:
    embeddings = json.load(f)

names = list(embeddings.keys())
vectors = [embeddings[name] for name in names]
vectors = np.array(vectors)

# Hitung cosine similarity
similarity_matrix = cosine_similarity(vectors)

# Buat DataFrame dan simpan
df = pd.DataFrame(similarity_matrix, index=names, columns=names)
df.to_csv("similarity_matrix.csv")
print("Cosine similarity matrix saved to similarity_matrix.csv")
