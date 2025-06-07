import os, json
import requests
from dotenv import load_dotenv
from typesense import Client

load_dotenv()

TYPESENSE_CLIENT = Client({
    "nodes": [{
        "host": os.getenv("TYPESENSE_HOST", "localhost"),
        "port": int(os.getenv("TYPESENSE_PORT", "8108")),
        "protocol": os.getenv("TYPESENSE_PROTOCOL", "http"),
    }],
    "api_key": os.getenv("TYPESENSE_API_KEY", "xyz"),
    "connection_timeout_seconds": 3,
})

NOMIC_API_KEY = os.getenv("NOMIC_API_KEY")
NOMIC_URL = "https://api-atlas.nomic.ai/v1/embedding/text"
NOMIC_MODEL = "nomic-embed-text-v1.5"

def embed_text(text: str):
    headers = {
        "Authorization": f"Bearer {NOMIC_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": NOMIC_MODEL,
        "input": [text]
    }
    resp = requests.post(NOMIC_URL, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()["embeddings"][0]  # Note: this key is correct per new API

def upload_chunks(file_path: str):
    with open(file_path) as f:
        chunks = json.load(f)

    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        url = chunk.get("url", "")

        if not text.strip():
            continue

        try:
            emb = embed_text(text)
            document = {
                "text": text,
                "url": url,
                "chunk": i,
                "embedding": emb
            }
            TYPESENSE_CLIENT.collections['tds_chunks'].documents.create(document)
            print(f"✅ Uploaded chunk {i}")
        except Exception as e:
            print(f"❌ Failed on chunk {i}: {e}")

upload_chunks("tds_chunks.json")  # Replace with your actual file
