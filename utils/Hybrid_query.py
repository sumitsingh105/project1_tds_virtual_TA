import os
import json
import base64
import requests
import numpy as np
import faiss
from PIL import Image
import pytesseract
import traceback
from rank_bm25 import BM25Okapi

# Constants
IMAGE_QUERY_PATH = "test1.png"
INDEX_PATH = "tds_faiss.index"
META_PATH = "tds_metadata.json"

JINA_API_KEY = os.getenv("JINA_API_KEY")
JINA_EMBED_URL = "https://api.jina.ai/v1/embeddings"
JINA_MODEL = "jina-clip-v2"

AIPIPE_API_KEY = os.getenv("AIPIPE_API_KEY")
EMBEDDING_URL = "https://aipipe.org/openai/v1/embeddings"
CHAT_URL = "https://aipipe.org/openrouter/v1/chat/completions"


def do_ocr(image_path):
    img = Image.open(image_path)
    return pytesseract.image_to_string(img).strip()


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
    return np.array(response.json()["data"][0]["embedding"], dtype=np.float32)



def dedupe_and_merge(faiss_results, bm25_results, max_results=5):
    seen_urls = set()
    merged = []

    def add_unique(results):
        for item in results:
            url = item.get("url") or item.get("source")
            if url and url not in seen_urls:
                seen_urls.add(url)
                merged.append(item)
            if len(merged) >= max_results:
                break

    add_unique(faiss_results)
    add_unique(bm25_results)

    return merged[:max_results]

def embed_with_jina_text(text):
    payload = {
        "model": JINA_MODEL,
        "input": [{"text": text}]
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
    return np.array(response.json()["data"][0]["embedding"], dtype=np.float32)


def search_faiss(query_vector, top_k=5):
    index = faiss.read_index(INDEX_PATH)
    print("📐 FAISS index dimension:", index.d)
    print("📐 Query vector shape:", query_vector.shape)

    with open(META_PATH, "r") as f:
        metadata = json.load(f)
    print("🗃️ Metadata entries:", len(metadata))

    distances, indices = index.search(np.expand_dims(query_vector, axis=0), top_k)
    print("📊 FAISS distances:", distances)
    print("📊 FAISS indices:", indices)

    return [metadata[i] for i in indices[0] if 0 <= i < len(metadata)]


def search_bm25(query_text, top_k=5):
    with open(META_PATH, "r") as f:
        metadata = json.load(f)

    corpus = [doc.get("text", "") for doc in metadata]
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    tokenized_query = query_text.lower().split()

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [metadata[i] for i in top_indices if scores[i] > 0]


def merge_results(faiss_results, bm25_results):
    seen = set()
    merged = []
    for item in faiss_results + bm25_results:
        uid = (item.get("text", ""), item.get("url", ""))
        if uid not in seen:
            seen.add(uid)
            merged.append(item)
    return merged[:7]  # trim if too many



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
You are an AI assistant that answers user questions based **only** on the provided context.

🔒 Rules:
- ONLY use the chunks below for your answer. Do not guess or use external knowledge.
- If the context doesn't answer the question, say so clearly in the `answer`.
- If you refer to any chunk, include its `url` and a short relevant `text` from that chunk in the `links`.

💡 Format:
You must return a valid JSON object like this:

{{
  "answer": "Your answer here.",
  "links": [
    {{
      "url": "https://example.com",
      "text": "short relevant text from that chunk"
    }},
    ...
  ]
}}

🧠 Context:
{joined_context}

❓ Question:
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
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 500
    }
    response = requests.post(CHAT_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def clean_response(raw_json_str):
    try:
        fixed = raw_json_str.replace("```json", "").replace("```", "").strip()
        return json.loads(fixed)
    except Exception:
        return {"answer": raw_json_str.strip(), "links": []}


def main():
    try:
        # === Step 1: Get query ===
        use_image = False  # Change this to False if using plain text

        if use_image:
            print("🔍 Performing OCR on image...")
            question_text = do_ocr(IMAGE_QUERY_PATH)
        else:
            question_text = input("Enter your text query: ").strip()

        if not question_text:
            raise ValueError("Query text is empty.")

        print("🔤 Query:", question_text)

        print("🎯 FAISS hits:", len(faiss_results))
        print("🎯 BM25 hits:", len(bm25_results))
        print("🎯 Combined hits:", len(combined_results))


        # === Step 2: Get text embedding (OpenAI/AIPipe) ===
        print("🧠 Getting text embedding...")
        query_vector = embed_with_jina_text(question_text)  # Jina embedding for FAISS search


        print("🎯 FAISS hits:", len(faiss_results))
        print("🎯 BM25 hits:", len(bm25_results))
        print("🎯 Combined hits:", len(combined_results))

        # === Step 3: Vector search ===
        print("📦 Searching vector index...")
        faiss_results = search_faiss(query_vector, top_k=5)



        print("🎯 FAISS hits:", len(faiss_results))
        print("🎯 BM25 hits:", len(bm25_results))
        print("🎯 Combined hits:", len(combined_results))

        # === Step 4: BM25 search ===
        print("🔍 Searching BM25 index...")
        bm25_results = search_bm25(question_text, top_k=5)
        print("🔍 BM25 Results:", bm25_results)

        print("🎯 FAISS hits:", len(faiss_results))
        print("🎯 BM25 hits:", len(bm25_results))
        print("🎯 Combined hits:", len(combined_results))

        # === Step 5: Combine ===
        print("⚙️ Combining results...")
        combined_results = dedupe_and_merge(faiss_results, bm25_results)

        if not combined_results:
            raise ValueError("No relevant data found for the query.")


        print("🎯 FAISS hits:", len(faiss_results))
        print("🎯 BM25 hits:", len(bm25_results))
        print("🎯 Combined hits:", len(combined_results))
        # === Step 6: Prompt + LLM ===
        prompt = build_prompt(question_text, combined_results)
        print("🤖 Sending to LLM...")
        raw_response = call_openai_chat_completion(prompt)

        final = clean_response(raw_response)
        print("✅ Final Response:")
        print(json.dumps(final, indent=2))

        print("🎯 FAISS hits:", len(faiss_results))
        print("🎯 BM25 hits:", len(bm25_results))
        print("🎯 Combined hits:", len(combined_results))

        try:
            print("📦 Searching vector index...")
            faiss_results = search_faiss(query_vector, top_k=5)
            print("🎯 FAISS hits:", len(faiss_results))
        except Exception as fe:
            print("❌ FAISS search failed:", fe)
            traceback.print_exc()
            faiss_results = []




if __name__ == "__main__":
    main()
