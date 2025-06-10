from __future__ import annotations
import os, json, traceback, re
from typing import List, Optional
from fastapi import File, UploadFile, Form
import numpy as np
import requests, typesense
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

# Load environment variables
load_dotenv()

# ENV CONFIG
TYPESENSE_API_KEY = os.getenv("TYPESENSE_API_KEY")
TYPESENSE_HOST = os.getenv("TYPESENSE_HOST")
TYPESENSE_PORT = int(os.getenv("TYPESENSE_PORT"))
TYPESENSE_PROTOCOL = os.getenv("TYPESENSE_PROTOCOL")
NOMIC_API_KEY = os.getenv("NOMIC_API_KEY")
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")

COLLECTION_NAME = "tds_chunks"
TOP_K = 5
NOMIC_URL = "https://api-atlas.nomic.ai/v1/embedding/text"
NOMIC_MODEL = "nomic-embed-text-v1.5"
AIPIPE_URL = "https://aipipe.org/openrouter/v1/chat/completions"
AIPIPE_MODEL = "google/gemini-2.0-flash-lite-001"

# FastAPI init
app = FastAPI(title="Hybrid RAG + AIpipe API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# Typesense init
typesense_client = typesense.Client({
    "nodes": [{
        "host": TYPESENSE_HOST,
        "port": TYPESENSE_PORT,
        "protocol": TYPESENSE_PROTOCOL
    }],
    "api_key": TYPESENSE_API_KEY,
    "connection_timeout_seconds": 10
})

class QueryRequest(BaseModel):
    question: str

# --- EMBEDDING ---
def embed_with_nomic(text: str) -> np.ndarray:
    headers = {
        "Authorization": f"Bearer {NOMIC_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": NOMIC_MODEL, "texts": [text]}
    response = requests.post(NOMIC_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    embedding = data.get("data", [{}])[0].get("embedding") or data.get("embeddings", [])[0]
    return np.array(embedding, dtype=np.float32)

# --- VECTOR SEARCH ---
def search_typesense_vector(query_vector, top_k=TOP_K):
    vector_str = ",".join(f"{v:.6f}" for v in query_vector.tolist())
    search_parameters = {
        "searches": [
            {
                "collection": COLLECTION_NAME,
                "q": "*",
                "query_by": "text",
                "vector_query": f"embedding:([{vector_str}], k:{top_k})"
            }
        ]
    }
    response = typesense_client.multi_search.perform(search_parameters)
    hits = response["results"][0]["hits"]

    print(f"\n[INFO] Retrieved {len(hits)} chunks from Typesense:")
    for i, hit in enumerate(hits):
        text = hit['document'].get("text", "")[:150].replace("\n", " ").strip()
        url = hit['document'].get("url", "N/A")
        print(f"Chunk {i+1}: {text}\n→ {url}\n")

    return hits

# --- PROMPT BUILDER ---
def build_prompt(user_q: str, chunks: List[dict]) -> str:
    if not chunks:
        return f"""
You are a helpful assistant. No relevant context was found for the following question:

{user_q}

Respond with: \"Sorry, I could not find any course content related to your question.\"
""".strip()

    context_texts = []
    for chunk in chunks:
        doc = chunk["document"]
        url = doc.get("url", "unknown")
        text = doc.get("text", "").replace("\n", " ").strip()
        context_texts.append(f"URL: {url}\nText: {text}")
    context = "\n\n---\n\n".join(context_texts)

    sample_response = {
        "answer": "You must use `gpt-3.5-turbo-0125`, even if the AI Proxy only supports `gpt-4o-mini`. Use the OpenAI API directly for this question.",
        "links": [
            {
                "url": "https://discourse.onlinedegree.iitm.ac.in/t/ga5-question-8-clarification/155939/4",
                "text": "Use the model that’s mentioned in the question."
            },
            {
                "url": "https://discourse.onlinedegree.iitm.ac.in/t/ga5-question-8-clarification/155939/3",
                "text": "My understanding is that you just have to use a tokenizer..."
            }
        ]
    }

    return f"""
You are a helpful AI assistant for data-science learners.

ONLY use the context below to answer the question. Do not make up facts. If the context doesn’t contain an answer, say so.

Return a JSON with:
- \"answer\": a helpful sentence based ONLY on context
- \"links\": exactly 2 helpful references (from the context, with URL + 1-line summary)

Context:
{context}

Question:
{user_q}

Respond with ONLY this JSON format:
{json.dumps(sample_response, indent=2)}
""".strip()

# --- JSON EXTRACTOR ---
def parse_llm_response(raw_text: str):
    pattern = r"```json\s*(\{.*?\})\s*```"
    match = re.search(pattern, raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(raw_text)
    except:
        return {"answer": raw_text.strip(), "links": []}

# --- MAIN API ---
@app.post("/api/")
async def rag_answer(
    request: Request,
    question: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    try:
        if request.headers.get("content-type", "").startswith("application/json"):
            data = await request.json()
            question = data.get("question", question)

        if not question and not image:
            raise HTTPException(status_code=400, detail="No question or image provided.")

        embedding = embed_with_nomic(question)
        chunks = search_typesense_vector(embedding, top_k=TOP_K)
        prompt = build_prompt(question, chunks)

        response = requests.post(
            AIPIPE_URL,
            headers={
                "Authorization": f"Bearer {AIPIPE_TOKEN}",
                "Content-Type": "application/json",
            },
            json={
                "model": AIPIPE_MODEL,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            },
            timeout=30
        )
        response.raise_for_status()
        raw_text = response.json()["choices"][0]["message"]["content"]

        return JSONResponse(content=parse_llm_response(raw_text))

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
