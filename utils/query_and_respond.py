import os
import json
import base64
import requests
import numpy as np
import faiss
from PIL import Image
import pytesseract
import traceback

# Constants
IMAGE_QUERY_PATH = "test.webp"
INDEX_PATH = "tds_faiss.index"
META_PATH = "tds_metadata.json"

JINA_API_KEY = "jina_f7dc2990880849c3b6c404d8e44cbe58u8AYLEAx4ZeXixBCCq4O6p4gnfgA"
JINA_EMBED_URL = "https://api.jina.ai/v1/embeddings"
JINA_MODEL = "jina-clip-v2"

AIPIPE_API_KEY = os.getenv("AIPIPE_API_KEY")
EMBEDDING_URL = "https://aipipe.org/openai/v1/embeddings"
CHAT_URL = "https://aipipe.org/openrouter/v1/chat/completions"


def do_ocr(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text.strip()


def read_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def embed_image(path):
    image_b64 = read_image_base64(path)
    payload = {
        "model": JINA_MODEL,
        "input": [{"image": image_b64}]
    }
    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(JINA_EMBED_URL, headers=headers, json=payload)
    response.raise_for_status()
    embedding = response.json()["data"][0]["embedding"]
    return np.array(embedding, dtype=np.float32)


def get_embedding(text):
    headers = {
        "Authorization": f"Bearer {AIPIPE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "text-embedding-3-small",
        "input": text
    }
    response = requests.post(EMBEDDING_URL, headers=headers, json=payload)
    response.raise_for_status()
    embedding = response.json()["data"][0]["embedding"]
    return np.array(embedding, dtype=np.float32)


def search_index(query_vector, top_k=5):
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r") as f:
        metadata = json.load(f)

    print(f"Searching with query vector shape: {query_vector.shape}")
    distances, indices = index.search(np.expand_dims(query_vector, axis=0), top_k)
    print(f"Distances: {distances}")
    print(f"indices: {indices}")

    results = []
    for i in indices[0]:
        if i == -1 or i >= len(metadata):
            continue
        results.append(metadata[i])
    return results



def build_prompt(question, retrieved_chunks):
    context_texts = []
    for chunk in retrieved_chunks:
        text = chunk.get("text", "").strip()
        url = chunk.get("url", "").strip()
        title = chunk.get("title", "").strip()
        if text:
            context_texts.append(f"[{title}]({url})\n{text}")

    joined_context = "\n\n---\n\n".join(context_texts)

    prompt = f"""
You are an AI assistant that answers user questions based **only** on the context provided below.

- Use only the provided chunks to answer.
- If you use information from any chunk, include it in the `links` section as a dictionary with `url` and short `text` from that chunk.
- Do not use general knowledge or guess beyond the chunks.
- Always respond in this JSON format:
{{
  "answer": "your concise answer based on the chunks",
  "links": [
    {{
      "url": "https://example.com",
      "text": "short summary from that chunk"
    }},
    ...
  ]
}}

Context:
{joined_context}

Question:
{question}
"""
    return prompt


def call_openai_chat_completion(prompt):
    headers = {
        "Authorization": f"Bearer {AIPIPE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }
    response = requests.post(CHAT_URL, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]
def clean_response(raw_json_str):
    try:
        fixed = raw_json_str.replace("```json", "").replace("```", "").strip()
        return json.loads(fixed)
    except Exception as e:
        return {
            "answer": raw_json_str.strip(),
            "links": []
        }


def main():
    try:
        print("Step 1: OCR starting")
        question_text = do_ocr(IMAGE_QUERY_PATH)
        print("OCR result:", question_text)

        print("Step 2: Get embedding")
        query_vector = get_embedding(question_text)  # or use get_embedding(question_text) if you want text embedding
        print("Embedding vector shape:", query_vector.shape)

        print("Step 3: Search FAISS")
        results = search_index(query_vector, top_k=5)
        print(f"Number of results found: {len(results)}")

        if not results:
            raise ValueError("No relevant data found for the query.")

        prompt = build_prompt(question_text, results)
        print("Prompt built, sending to AI Pipe chat completion...")

        raw_response = call_openai_chat_completion(prompt)
        parsed = clean_response(raw_response)
        print(json.dumps(parsed, indent=2))



    except Exception as e:
        print("❌ Error during processing:", e)
        traceback.print_exc()


if __name__ == "__main__":
    main()
