import os
import json
import requests
import numpy as np
import faiss
from tqdm import tqdm

# Input & output file paths
INPUT_JSON = "combined_tds_rag_data.json"
FAISS_INDEX_PATH = "tds_faiss.index"
METADATA_PATH = "tds_metadata.json"

# AI Pipe / OpenAI API details
AIPIPE_API_KEY = os.getenv("AIPIPE_API_KEY")
EMBEDDING_URL = "https://aipipe.org/openai/v1/embeddings"
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 20  # Batch size for embedding API

# Load data
with open(INPUT_JSON, "r") as f:
    raw_data = json.load(f)

# Clean up and filter chunks
chunks = [d for d in raw_data if d.get("text", "").strip()]
print(f"✅ Loaded {len(chunks)} valid text chunks")

# Batched embedder
def embed_batch(texts):
    headers = {
        "Authorization": f"Bearer {AIPIPE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": EMBEDDING_MODEL,
        "input": texts
    }
    response = requests.post(EMBEDDING_URL, headers=headers, json=payload)
    response.raise_for_status()
    embeddings = [item["embedding"] for item in response.json()["data"]]
    return embeddings

# Process in batches
all_embeddings = []
all_metadata = []

print("🔍 Generating embeddings in batches...")
for i in tqdm(range(0, len(chunks), BATCH_SIZE)):
    batch = chunks[i:i + BATCH_SIZE]
    texts = [x["text"] for x in batch]
    try:
        embeddings = embed_batch(texts)
        all_embeddings.extend(embeddings)
        all_metadata.extend(batch)
    except Exception as e:
        print(f"⚠️ Batch {i}-{i+BATCH_SIZE} failed: {e}")

# Convert and save
embedding_matrix = np.array(all_embeddings, dtype=np.float32)
dimension = embedding_matrix.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embedding_matrix)

faiss.write_index(index, FAISS_INDEX_PATH)
with open(METADATA_PATH, "w") as f:
    json.dump(all_metadata, f)

print(f"✅ Saved {len(all_embeddings)} embeddings to FAISS index")
print(f"✅ Metadata saved to {METADATA_PATH}")
