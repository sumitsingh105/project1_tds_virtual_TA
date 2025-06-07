import os
import base64
import traceback
import json
from typing import Optional, List
import faiss
from rank_bm25 import BM25Okapi


import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
import pytesseract
from dotenv import load_dotenv
load_dotenv()

# Constants
IMAGE_QUERY_PATH = "/Users/sumitsingh/Desktop/IIT_Madras/TDS/Project_1/test1.png"
INDEX_PATH = "/Users/sumitsingh/Desktop/IIT_Madras/TDS/Project_1/tds_faiss.index"

META_PATH = "/Users/sumitsingh/Desktop/IIT_Madras/TDS/Project_1/tds_metadata.json"

JINA_API_KEY = os.getenv("JINA_API_KEY")
JINA_EMBED_URL = "https://api.jina.ai/v1/embeddings"
JINA_MODEL = "jina-clip-v2"

AIPIPE_API_KEY = os.getenv("AIPIPE_API_KEY")
EMBEDDING_URL = "https://aipipe.org/openai/v1/embeddings"
CHAT_URL = "https://aipipe.org/openrouter/v1/chat/completions"

app = FastAPI(title="Data Science Virtual TA API")

# Allow CORS for all origins for local testing (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)



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
    with open(META_PATH, "r") as f:
        metadata = json.load(f)

    distances, indices = index.search(np.expand_dims(query_vector, axis=0), top_k)
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


import json

def build_prompt(query_text, top_chunks):
    # Use the top relevant chunk (you can change this logic to include multiple chunks if needed)
    top_chunk = top_chunks[0]

    # Prepare a sample link to guide the LLM
    sample_link = {
        "url": top_chunk.get("url", "https://tds.s-anand.net/"),  # Fallback if URL missing
        "text": top_chunk.get("text", "")[:160]  # First 160 characters for context
    }

    # Prepare example response as JSON string
    example_response = {
        "answer": "your answer here",
        "links": [sample_link]
    }

    # Now, format the actual prompt
    prompt = f"""
You are a helpful educational assistant for data science learners. Your task is to answer the user’s query using only the provided context. Also provide related links.

Rules:
- Use only the provided context.
- Do not invent facts or refer to external sources.
- Format your response as a JSON object with:
  - "answer": the answer to the question
  - "links": a list of {{"url", "text"}} pairs from relevant context
- DO NOT include chunk numbers in URLs (e.g., strip off any #chunk1).

User Query:
{query_text}

Context Chunks:
{json.dumps(top_chunks, indent=2)}

Respond in the following JSON format:
{json.dumps(example_response, indent=2)}
"""
    return prompt.strip()


def call_openai_chat_completion(prompt):
    headers = {
        "Authorization": f"Bearer {AIPIPE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-3.5-turbo",
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

class QueryRequest(BaseModel):
    question: Optional[str] = "what is github"
    images_base64: Optional[List[str]] = None  # list of base64-encoded images

def embed_with_jina_text(text: str) -> np.ndarray:
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

# Define your other helper functions here like:
# search_faiss, search_bm25, dedupe_and_merge, build_prompt, call_openai_chat_completion, clean_response

@app.post("/api/")
async def answer_question(req: QueryRequest):
    try:
        question_text = None

        # If base64 images provided, do OCR on first image
        if req.images_base64 and len(req.images_base64) > 0:
            img_data = base64.b64decode(req.images_base64[0])
            img = Image.open(BytesIO(img_data))
            question_text = pytesseract.image_to_string(img).strip()

            if not question_text:
                raise HTTPException(status_code=400, detail="OCR failed to extract text from image.")
        else:
            question_text = req.question

        if not question_text or question_text.strip() == "":
            raise HTTPException(status_code=400, detail="No query text provided.")

        question_text = question_text.strip()

        # Embedding, search, merging results
        query_vector = embed_with_jina_text(question_text)
        faiss_results = search_faiss(query_vector, top_k=5)
        bm25_results = search_bm25(question_text, top_k=5)
        combined_results = dedupe_and_merge(faiss_results, bm25_results)

        if not combined_results:
            return {"answer": "No relevant information found for your query.", "links": []}

        prompt = build_prompt(question_text, combined_results)
        raw_response = call_openai_chat_completion(prompt)
        final_response = clean_response(raw_response)

        return final_response

    except HTTPException as he:
        raise he
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"message": "API is running. Use POST /api/ to query."}

@app.get("/api/")
def api_get():
    return {"message": "This endpoint accepts POST requests with a JSON body."}
