import typesense
from typesense.exceptions import ObjectNotFound  # <-- explicit import

# ---- 1. connect -------------------------------------------------------------
client = typesense.Client(
    {
        "nodes": [
            {"host": "localhost", "port": "8108", "protocol": "http"},
        ],
        "api_key": "xyz",          # TODO: replace in production
        "connection_timeout_seconds": 2,
    }
)

# ---- 2. define schema -------------------------------------------------------
EMBED_DIM = 768  # set to the length of your embeddings

schema = {
    "name": "tds_chunks",
    "fields": [
        {"name": "text",  "type": "string"},
        {"name": "url",   "type": "string"},
        {"name": "chunk", "type": "int32"},
        {
            "name":   "embedding_vector",
            "type":   "float[]",
            "num_dim": EMBED_DIM,   # <-- required
        },
    ],
    "default_sorting_field": "chunk",
}

# ---- 3. (re)create collection ----------------------------------------------
try:
    client.collections["tds_chunks"].delete()
    print("Old collection deleted.")
except ObjectNotFound:
    print("Collection did not exist; continuing.")
except Exception as exc:
    print(f"Could not delete collection: {exc}")

collection = client.collections.create(schema)
print(f"Collection `{collection['name']}` created.")

# ---- 4. sanity-check --------------------------------------------------------
print("Current collections:")
for coll in client.collections.retrieve():
    print("-", coll["name"])
