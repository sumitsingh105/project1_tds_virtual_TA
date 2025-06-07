import os
import json
import base64
import requests
import numpy as np
import faiss
from tqdm import tqdm

COMBINED_DATA_PATH = "combined_tds_rag_data.json"
INDEX_PATH = "tds_faiss.index"
META_PATH = "tds_metadata.json"

JINA_API_KEY = os.getenv("JINA_API_KEY")
JINA_EMBED_URL = "https://api.jina.ai/v1/embeddings"
JINA_MODEL = "jina-clip-v2"
BATCH_SIZE = 16  # Tune this based on payload size limits

def read_image_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def build_jina_input(entry):
    inputs = []
    types = []
    image_paths = []

    if "text" in entry and entry["text"]:
        inputs.append({"text": entry["text"]})
        types.append("text")
        image_paths.append(None)

    for image_path in entry.get("images", []):
        if os.path.exists(image_path):
            inputs.append({"image": read_image_base64(image_path)})
            types.append("image")
            image_paths.append(image_path)
        else:
            print(f"[Warning] Missing image: {image_path}")

    return inputs, types, image_paths

def get_embeddings(batch_input):
    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": JINA_MODEL,
        "input": batch_input
    }
    response = requests.post(JINA_EMBED_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["data"]

def main():
    with open(COMBINED_DATA_PATH, "r") as f:
        data = json.load(f)

    all_embeddings = []
    all_metadata = []

    batch_inputs = []
    batch_context = []

    for entry in tqdm(data, desc="Preparing batches"):
        inputs, types, image_paths = build_jina_input(entry)
        if not inputs:
            continue

        for i, item in enumerate(inputs):
            batch_inputs.append(item)
            batch_context.append({
                "id": entry.get("id", ""),
                "type": types[i],
                "source": entry.get("url", ""),
                "text": entry.get("text", "") if types[i] == "text" else "",
                "image_path": image_paths[i] if types[i] == "image" else None
            })

        # Send batch
        if len(batch_inputs) >= BATCH_SIZE:
            try:
                embeddings = get_embeddings(batch_inputs)
                for emb, meta in zip(embeddings, batch_context):
                    all_embeddings.append(np.array(emb["embedding"], dtype=np.float32))
                    all_metadata.append(meta)
            except Exception as e:
                print(f"[Error] Embedding batch failed: {e}")
            batch_inputs = []
            batch_context = []

    # Final remaining
    if batch_inputs:
        try:
            embeddings = get_embeddings(batch_inputs)
            for emb, meta in zip(embeddings, batch_context):
                all_embeddings.append(np.array(emb["embedding"], dtype=np.float32))
                all_metadata.append(meta)
        except Exception as e:
            print(f"[Error] Final embedding batch failed: {e}")

    if not all_embeddings:
        print("❌ No embeddings found.")
        return

    index = faiss.IndexFlatL2(len(all_embeddings[0]))
    index.add(np.stack(all_embeddings))
    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "w") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"[✅] Saved {len(all_embeddings)} embeddings to {INDEX_PATH}")
    print(f"[✅] Metadata saved to {META_PATH}")

if __name__ == "__main__":
    main()
