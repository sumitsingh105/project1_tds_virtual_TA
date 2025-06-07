import os
import json
import base64
import requests
import numpy as np
import faiss
import pickle
from tqdm import tqdm
from rank_bm25 import BM25Okapi

# Paths
COMBINED_DATA_PATH = "combined_tds_rag_data1.json"
FAISS_INDEX_PATH = "tds_faiss.index"
META_PATH = "tds_metadata.json"
BM25_INDEX_PATH = "tds_bm25.pkl"

# Jina config
JINA_API_KEY = os.getenv("JINA_API_KEY")
JINA_EMBED_URL = "https://api.jina.ai/v1/embeddings"
JINA_MODEL = "jina-clip-v2"
BATCH_SIZE = 16

def read_image_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def build_jina_input(entry):
    inputs = []
    types = []
    image_paths = []

    if entry.get("text"):
        inputs.append({"text": entry["text"]})
        types.append("text")
        image_paths.append(None)

    for image_path in entry.get("images", []):
        if os.path.exists(image_path):
            inputs.append({"image": read_image_base64(image_path)})
            types.append("image")
            image_paths.append(image_path)
        else:
            print(f"[⚠️] Missing image: {image_path}")
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

def build_bm25_index(corpus):
    print("📚 Building BM25 index...")
    tokenized = [text.lower().split() for text in corpus]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)
    print(f"✅ BM25 index saved to: {BM25_INDEX_PATH}")

def main():
    with open(COMBINED_DATA_PATH, "r") as f:
        data = json.load(f)

    all_embeddings = []
    all_metadata = []
    bm25_corpus = []

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

        if len(batch_inputs) >= BATCH_SIZE:
            try:
                embeddings = get_embeddings(batch_inputs)
                for emb, meta in zip(embeddings, batch_context):
                    all_embeddings.append(np.array(emb["embedding"], dtype=np.float32))
                    all_metadata.append(meta)
                    if meta["type"] == "text":
                        bm25_corpus.append(meta["text"])
            except Exception as e:
                print(f"[❌] Embedding batch failed: {e}")
            batch_inputs = []
            batch_context = []

    # Final batch
    if batch_inputs:
        try:
            embeddings = get_embeddings(batch_inputs)
            for emb, meta in zip(embeddings, batch_context):
                all_embeddings.append(np.array(emb["embedding"], dtype=np.float32))
                all_metadata.append(meta)
                if meta["type"] == "text":
                    bm25_corpus.append(meta["text"])
        except Exception as e:
            print(f"[❌] Final embedding batch failed: {e}")

    if not all_embeddings:
        print("❌ No embeddings were created.")
        return

    print("🧠 Saving FAISS index...")
    index = faiss.IndexFlatL2(len(all_embeddings[0]))
    index.add(np.stack(all_embeddings))
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"✅ FAISS index saved to: {FAISS_INDEX_PATH}")

    print("💾 Saving metadata...")
    with open(META_PATH, "w") as f:
        json.dump(all_metadata, f, indent=2)
    print(f"✅ Metadata saved to: {META_PATH}")

    build_bm25_index(bm25_corpus)

if __name__ == "__main__":
    main()
