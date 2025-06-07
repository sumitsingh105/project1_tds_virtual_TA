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
from sentence_transformers import SentenceTransformer

from PIL import Image
from io import BytesIO
import pytesseract
from dotenv import load_dotenv

# ---------------------------------------------------------
# 1. Load environment variables
# ---------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------
# 2. Constants for FAISS / metadata / Jina embeddings
# ---------------------------------------------------------
INDEX_PATH    = "/Users/sumitsingh/Desktop/IIT_Madras/TDS/Project_1/tds_faiss.index"
META_PATH     = "/Users/sumitsingh/Desktop/IIT_Madras/TDS/Project_1/tds_metadata.json"
JINA_API_KEY  = os.getenv("JINA_API_KEY")
JINA_EMBED_URL= "https://api.jina.ai/v1/embeddings"
JINA_MODEL    = "jina-clip-v2"

# ---------------------------------------------------------
# 3. Constants for Ollama
# ---------------------------------------------------------
OLLAMA_URL    = "http://localhost:11434/v1/chat/completions"
OLLAMA_MODEL  = "gemma3:1b-it-qat"  # adjust if your model name differs

# ---------------------------------------------------------
# 4. FastAPI app setup
# ---------------------------------------------------------
app = FastAPI(title="Data Science Virtual TA API (using Ollama)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# 5. Pydantic request/response schemas
# ---------------------------------------------------------
class QueryRequest(BaseModel):
    question: Optional[str] = None
    image: Optional[str] = None  # Base64 image string

class Link(BaseModel):
    url: str
    text: str

# We’ll always return at least an “answer” (string) and a list of zero or more “links”
class AnswerResponse(BaseModel):
    answer: str
    links: List[Link]

# ---------------------------------------------------------
# 6. Utility functions
# ---------------------------------------------------------
def strip_chunk(url: str) -> str:
    return url.split('#')[0]

def embed_with_nomic_text(text: str) -> np.ndarray:
    """
    Generates embedding using Nomic Atlas API (v1.5 model).
    Compatible with OpenAI-style embedding.
    """
    import os
    import requests
    import numpy as np

    NOMIC_URL = "https://api-atlas.nomic.ai/v1/embedding/text"
    NOMIC_API_KEY = os.getenv("NOMIC_API_KEY")
    
    headers = {
        "Authorization": f"Bearer {NOMIC_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "nomic-embed-text-v1.5",  # ✅ updated to latest model
        "input": [text]
    }

    response = requests.post(NOMIC_URL, headers=headers, json=payload)
    response.raise_for_status()

    embedding = response.json()["data"][0]["embedding"]
    return np.array(embedding, dtype=np.float32)


def search_faiss(query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r") as f:
        metadata = json.load(f)
    distances, indices = index.search(np.expand_dims(query_vector, axis=0), top_k)
    return [metadata[i] for i in indices[0] if 0 <= i < len(metadata)]

def search_bm25(query_text: str, top_k: int = 5) -> list[dict]:
    with open(META_PATH, "r") as f:
        metadata = json.load(f)
    corpus = [doc.get("text", "") for doc in metadata]
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    tokenized_query  = query_text.lower().split()
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [metadata[i] for i in top_indices if scores[i] > 0]

def dedupe_and_merge(faiss_results: list[dict], bm25_results: list[dict], max_results: int = 5) -> list[dict]:
    seen_urls = set()
    merged    = []

    def add_unique(results: list[dict]):
        nonlocal merged, seen_urls
        for item in results:
            raw_url = item.get("url") or item.get("source", "")
            if raw_url:
                clean_url = strip_chunk(raw_url)
                if clean_url not in seen_urls:
                    seen_urls.add(clean_url)
                    item["url"] = clean_url
                    merged.append(item)
            if len(merged) >= max_results:
                break

    add_unique(faiss_results)
    add_unique(bm25_results)
    return merged[:max_results]

def build_prompt(query_text: str, top_chunks: list[dict]) -> str:
    top_chunk = top_chunks[0]
    sample_link = {
        "url": top_chunk.get("url", "https://tds.s-anand.net/"),
        "text": top_chunk.get("text", "")[:160]
    }
    example_response = {"answer": "your answer here", "links": [sample_link]}
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

# ---------------------------------------------------------
# 7. Ollama chat completion function
# ---------------------------------------------------------
def call_ollama_chat_completion(prompt: str) -> str:
    """
    Sends prompt to Ollama’s local HTTP server and returns the model’s reply.
    Expects '{"model":"<model_name>","messages":[{"role":"user","content":"<prompt>"}]}'.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    # Ollama returns {"choices":[{"message":{"role":"assistant","content":"…"}}], …}
    choices = data.get("choices", [])
    if isinstance(choices, list) and len(choices) > 0:
        return choices[0].get("message", {}).get("content", "").strip()
    return ""

# ---------------------------------------------------------
# 8. Clean and parse the raw LLM response
# ---------------------------------------------------------
def clean_response(raw_json_str: str) -> dict:
    """
    The prompt expects the LLM to respond with a JSON blob containing "answer" and "links".
    We strip ```json fences, parse JSON, then sanitize keys.
    """
    try:
        fixed = raw_json_str.replace("```json", "").replace("```", "").strip()
        result = json.loads(fixed)

        answer = result.get("answer")
        links  = result.get("links", [])

        if not isinstance(answer, str):
            # If JSON didn't contain "answer", fall back to returning the raw string
            answer = fixed

        clean_links = []
        if isinstance(links, list):
            for link in links:
                if isinstance(link, dict):
                    url  = link.get("url")
                    text = link.get("text")
                    if isinstance(url, str) and isinstance(text, str):
                        url = strip_chunk(url)
                        clean_links.append({"url": url, "text": text})
        return {"answer": answer, "links": clean_links}

    except Exception:
        # If JSON parsing fails, return raw text as "answer"
        return {"answer": raw_json_str.strip(), "links": []}

# ---------------------------------------------------------
# 9. Main /api/ endpoint
# ---------------------------------------------------------
@app.post("/api/", response_model=AnswerResponse)
async def answer_question(req: QueryRequest):
    try:
        # 9.1 Extract or OCR‐decode the question text
        question_text: Optional[str] = None

        if req.image:
            img_data = base64.b64decode(req.image)
            img = Image.open(BytesIO(img_data))
            question_text = pytesseract.image_to_string(img).strip()
            if not question_text:
                raise HTTPException(status_code=400, detail="OCR failed to extract text from image.")
        else:
            question_text = req.question

        if not question_text or question_text.strip() == "":
            raise HTTPException(status_code=400, detail="No query text provided.")

        question_text = question_text.strip()

        # 9.2 Perform embedding + retrieval (FAISS + BM25)
        query_vector  = embed_with_jina_text(question_text)
        faiss_results = search_faiss(query_vector, top_k=5)
        bm25_results  = search_bm25(question_text, top_k=5)
        combined_results = dedupe_and_merge(faiss_results, bm25_results)

        # 9.3 If no relevant context found, return fallback
        if not combined_results:
            return {"answer": "Sorry, this information is not available in the provided context.", "links": []}

        # 9.4 Build a RAG‐style prompt
        prompt = build_prompt(question_text, combined_results)

        # 9.5 Send that prompt to Ollama and get raw reply
        raw_response = call_ollama_chat_completion(prompt)

        # 9.6 Parse & sanitize into {"answer": str, "links": [ {url,text}, … ]}
        final_response = clean_response(raw_response)

        # 9.7 Guarantee the keys exist and have correct types
        if "answer" not in final_response or not isinstance(final_response["answer"], str):
            final_response["answer"] = "Sorry, I could not generate a valid answer."
        if "links" not in final_response or not isinstance(final_response["links"], list):
            final_response["links"] = []

        return final_response

    except HTTPException as he:
        raise he
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------
# 10. Health check route (optional)
# ---------------------------------------------------------
@app.get("/")
def root():
    return {"message": "API is running. Use POST /api/ to query."}
