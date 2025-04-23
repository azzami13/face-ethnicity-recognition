from deepface import DeepFace
import os
import json

input_dir = "../processed_images"
output_path = "embeddings.json"

embeddings = {}

for fname in os.listdir(input_dir):
    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(input_dir, fname)
        try:
            result = DeepFace.represent(img_path=path, model_name='ArcFace', enforce_detection=False)
            embeddings[fname] = result[0]["embedding"]
            print(f"Success: {fname}")
        except Exception as e:
            print(f"Failed on {fname}: {e}")

# Simpan embedding ke file JSON
with open(output_path, "w") as f:
    json.dump(embeddings, f)
print(f"Embeddings saved to {output_path}")
