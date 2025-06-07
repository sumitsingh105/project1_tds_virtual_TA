import json
import os
import faiss
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from sentence_transformers.util import normalize_embeddings

client = OpenAI()

# Path to your combined JSON file
COMBINED_DATA_PATH = "combined_rag_data.json"
INDEX_PATH = "tds_faiss.index"
META_PATH = "tds_metadata.json"

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_embeddings(texts):
    CHUNK_SIZE = 100
    all_embeddings = []

    for i in tqdm(range(0, len(texts), CHUNK_SIZE), desc="Embedding chunks"):
        batch = texts[i:i + CHUNK_SIZE]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        vectors = [d.embedding for d in response.data]
        all_embeddings.extend(vectors)
    return np.array(all_embeddings, dtype="float32")

def main():
    data = load_data(COMBINED_DATA_PATH)
    texts = [item["text"] for item in data]

    print("🔢 Generating embeddings...")
    embeddings = get_embeddings(texts)
    embeddings = normalize_embeddings(embeddings)  # improves similarity results

    print("🧠 Building FAISS index...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    print(f"✅ FAISS index saved to: {INDEX_PATH}")

    # Save metadata (includes source, url, etc.)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ Metadata saved to: {META_PATH}")

if __name__ == "__main__":
    main()
