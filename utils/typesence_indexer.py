"""
Bulk-index TDS chunks into Typesense with Nomic Atlas embeddings.
"""

import os, json, time, typesense, re
from dotenv import load_dotenv
from typesense.exceptions import ObjectNotFound, ObjectUnprocessable
from nomic import login, embed as nomic_embed

# ----------------------------------------------------------------------------
# 1. Config & sanity checks
# ----------------------------------------------------------------------------
load_dotenv()
NOMIC_API_KEY     = os.getenv("NOMIC_API_KEY")
TYPESENSE_API_KEY = os.getenv("TYPESENSE_API_KEY")
if not (NOMIC_API_KEY and TYPESENSE_API_KEY):
    raise RuntimeError("Missing NOMIC_API_KEY or TYPESENSE_API_KEY in .env")

EMBED_DIM        = 768
COLLECTION_NAME  = "tds_chunks"
DATA_FILE        = "/Users/sumitsingh/Desktop/IIT_Madras/TDS/Project_1/tds_scraper/combined_tds_rag_data1.json"
BATCH_SIZE       = 16
MAX_TEXT_LENGTH  = 5000  # To avoid Typesense payload limit

# ----------------------------------------------------------------------------
# 2. Connect to Typesense
# ----------------------------------------------------------------------------
client = typesense.Client({
    "nodes": [{"host": "localhost", "port": "8108", "protocol": "http"}],
    "api_key": TYPESENSE_API_KEY,
    "connection_timeout_seconds": 10,
})

try:
    client.collections[COLLECTION_NAME].delete()
    print("Old collection deleted.")
except ObjectNotFound:
    print("No existing collection found.")

schema = {
    "name": COLLECTION_NAME,
    "fields": [
        {"name": "id",        "type": "string"},
        {"name": "title",     "type": "string", "optional": True},
        {"name": "source",    "type": "string", "optional": True},
        {"name": "text",      "type": "string"},
        {"name": "url",       "type": "string"},
        {"name": "embedding", "type": "float[]", "num_dim": EMBED_DIM},
    ]
}
client.collections.create(schema)
print("Collection created.")

# ----------------------------------------------------------------------------
# 3. Embedding function using Nomic
# ----------------------------------------------------------------------------
login(NOMIC_API_KEY)

import time
from nomic import embed as nomic_embed
import requests

def get_embeddings(texts, retries=3, delay=5):
    for attempt in range(retries):
        try:
            result = nomic_embed.text(
                texts=texts,
                model="nomic-embed-text-v1.5"
            )
            return result["embeddings"]
        except Exception as e:
            if "429" in str(e):
                print(f"⚠️ Rate limited. Sleeping for {delay} seconds (attempt {attempt+1}/{retries})...")
                time.sleep(delay)
                delay *= 2  # exponential backoff
            else:
                raise
    raise RuntimeError("❌ Failed after multiple retries due to rate limiting.")


# ----------------------------------------------------------------------------
# 4. Sanitize Typesense IDs
# ----------------------------------------------------------------------------
def sanitize_id(raw_id: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_-]', '_', raw_id)

# ----------------------------------------------------------------------------
# 5. Load data and index in batches
# ----------------------------------------------------------------------------
with open(DATA_FILE, encoding="utf-8") as fh:
    docs = json.load(fh)

for start in range(0, len(docs), BATCH_SIZE):
    batch = docs[start : start + BATCH_SIZE]
    embeddings = get_embeddings([d["text"][:MAX_TEXT_LENGTH] for d in batch])

    lines = []
    for idx, (d, vec) in enumerate(zip(batch, embeddings)):
        original_id = d.get("id", f"{start+idx}")
        doc_id = sanitize_id(original_id)

        try:
            # Truncate text and validate embedding
            truncated_text = d["text"][:MAX_TEXT_LENGTH]
            assert isinstance(vec, list), f"Embedding is not a list: {vec}"
            assert len(vec) == EMBED_DIM, f"Embedding length mismatch: {len(vec)}"
            assert all(isinstance(x, (float, int)) for x in vec), "Embedding contains non-float values"

            lines.append(
                json.dumps({
                    "id":        doc_id,
                    "title":     d.get("title", ""),
                    "source":    d.get("source", ""),
                    "text":      truncated_text,
                    "url":       d["url"],
                    "embedding": vec,
                }, ensure_ascii=False)
            )
        except Exception as e:
            print(f"⚠️ Skipping doc {original_id}: {e}")
            continue

    if not lines:
        print(f"⚠️ No valid docs in batch {start}-{start+len(batch)-1}, skipping indexing.")
        continue

    ndjson = "\n".join(lines)

    try:
        raw_result = client.collections[COLLECTION_NAME].documents.import_(
            ndjson.encode("utf-8"),
            {"action": "upsert", "batch_size": BATCH_SIZE}
        )
        result = [json.loads(line) for line in raw_result.strip().split("\n")]
        failures = [r for r in result if not r["success"]]
        if failures:
            raise ObjectUnprocessable(failures)
    except Exception as e:
        print(f"❌ Batch {start}-{start+len(batch)-1} failed: {e}")
    else:
        print(f"✓ Indexed docs {start}-{start+len(batch)-1}")

print("Indexing complete 🎉")


embeddings = get_embeddings(texts)

print(f"Number of embeddings: {len(embeddings)}")
print(f"Embedding vector shape (dim): {len(embeddings[0])}")

# Check types
print(type(embeddings))        # Should be list
print(type(embeddings[0]))     # Should be list or np.array
print(all(isinstance(x, (float, int)) for x in embeddings[0]))  # Should be True
