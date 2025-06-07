# File: typesense_indexer_range.py

import os, json, time, typesense, re, sys
from dotenv import load_dotenv
from typesense.exceptions import ObjectNotFound, ObjectUnprocessable
from nomic import login, embed as nomic_embed

# ----------------------------------------------------------------------------
# 1. Config
# ----------------------------------------------------------------------------
load_dotenv()
NOMIC_API_KEY     = os.getenv("NOMIC_API_KEY")
TYPESENSE_API_KEY = os.getenv("TYPESENSE_API_KEY")
if not (NOMIC_API_KEY and TYPESENSE_API_KEY):
    raise RuntimeError("Missing NOMIC_API_KEY or TYPESENSE_API_KEY in .env")

EMBED_DIM         = 768
COLLECTION_NAME   = "tds_chunks"
DATA_FILE         = "./combined_tds_rag_data2.json"
BATCH_SIZE        = 32
MAX_TEXT_LENGTH   = 5000

# Parse range from command-line args
if len(sys.argv) != 3:
    print("Usage: python typesense_indexer_range.py <start_index> <end_index>")
    sys.exit(1)

START_INDEX = int(sys.argv[1])
END_INDEX   = int(sys.argv[2])

# ----------------------------------------------------------------------------
# 2. Connect to Typesense
# ----------------------------------------------------------------------------
client = typesense.Client({
    "nodes": [{"host": "localhost", "port": "8108", "protocol": "http"}],
    "api_key": TYPESENSE_API_KEY,
    "connection_timeout_seconds": 10,
})

# Only create schema if this is the first chunk
if START_INDEX == 0:
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
# 3. Embedding
# ----------------------------------------------------------------------------
login(NOMIC_API_KEY)

def get_embeddings(texts, retries=3, delay=5):
    for attempt in range(retries):
        try:
            result = nomic_embed.text(texts=texts, model="nomic-embed-text-v1.5")
            return result["embeddings"]
        except Exception as e:
            if "429" in str(e):
                print(f"⚠️ Rate limited. Sleeping for {delay} seconds (attempt {attempt+1}/{retries})...")
                time.sleep(delay)
                delay *= 2
            else:
                raise
    raise RuntimeError("❌ Failed after multiple retries due to rate limiting.")

def sanitize_id(raw_id: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_-]', '_', raw_id)

# ----------------------------------------------------------------------------
# 4. Load data and index
# ----------------------------------------------------------------------------
with open(DATA_FILE, encoding="utf-8") as fh:
    docs = json.load(fh)

batch_docs = docs[START_INDEX:END_INDEX]
print(f"Indexing docs {START_INDEX} to {END_INDEX}...")

for start in range(0, len(batch_docs), BATCH_SIZE):
    batch = batch_docs[start:start + BATCH_SIZE]
    try:
        embeddings = get_embeddings([d["text"][:MAX_TEXT_LENGTH] for d in batch])
    except RuntimeError as e:
        print(f"❌ Skipping batch {start}: {e}")
        continue

    lines = []
    for idx, (d, vec) in enumerate(zip(batch, embeddings)):
        original_id = d.get("id", f"{START_INDEX+start+idx}")
        doc_id = sanitize_id(original_id)

        lines.append(json.dumps({
            "id": doc_id,
            "title": d.get("title", ""),
            "source": d.get("source", ""),
            "text": d["text"][:MAX_TEXT_LENGTH],
            "url": d["url"],
            "embedding": vec,
        }, ensure_ascii=False))

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
        print(f"❌ Batch {START_INDEX+start}-{START_INDEX+start+len(batch)-1} failed: {e}")
    else:
        print(f"✓ Indexed docs {START_INDEX+start}-{START_INDEX+start+len(batch)-1}")

print("✅ Finished range:", START_INDEX, "to", END_INDEX)
