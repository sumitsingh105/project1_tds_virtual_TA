from __future__ import annotations
import os, json, traceback
from typing import List, Optional

import numpy as np
import requests, typesense
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

TYPESENSE_API_KEY = os.getenv("TYPESENSE_API_KEY", "xyz")
TYPESENSE_HOST = os.getenv("TYPESENSE_HOST", "localhost")
TYPESENSE_PORT = int(os.getenv("TYPESENSE_PORT", "8108"))
TYPESENSE_PROTOCOL = os.getenv("TYPESENSE_PROTOCOL", "http")

NOMIC_API_KEY = os.getenv("NOMIC_API_KEY")
NOMIC_URL = "https://api-atlas.nomic.ai/v1/embedding/text"
NOMIC_MODEL = "nomic-embed-text-v1.5"

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
OLLAMA_MODEL = "gemma3:1b-it-qat"

EMBED_DIM = 768
COLLECTION_NAME = "tds_chunks"
TOP_K = 5

app = FastAPI(title="Typesense Vector RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

client = typesense.Client({
    "nodes": [{
        "host": TYPESENSE_HOST,
        "port": TYPESENSE_PORT,
        "protocol": TYPESENSE_PROTOCOL,
    }],
    "api_key": TYPESENSE_API_KEY,
    "connection_timeout_seconds": 3,
})

class QueryRequest(BaseModel):
    question: Optional[str] = None
    images_base64: Optional[List[str]] = None

def embed_with_nomic(text: str) -> np.ndarray:
    if not NOMIC_API_KEY:
        raise RuntimeError("NOMIC_API_KEY is not set in environment variables.")
    headers = {
        "Authorization": f"Bearer {NOMIC_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": NOMIC_MODEL,
        "texts": [text]
    }
    response = requests.post(NOMIC_URL, headers=headers, json=payload, timeout=30)
    if response.status_code != 200:
        print(f"Nomic API error {response.status_code}: {response.text}")
        response.raise_for_status()
    response_json = response.json()
    
    if "data" in response_json:
        embedding = response_json["data"][0]["embedding"]
    elif "embeddings" in response_json:
        embedding = response_json["embeddings"][0]
    else:
        raise RuntimeError(f"Unexpected response format: {response_json}")
    
    return np.array(embedding, dtype=np.float32)

def search_typesense_vector(query_vector, top_k=TOP_K):
    try:
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
        response = client.multi_search.perform(search_parameters)
        hits = response["results"][0]["hits"]
        return hits
    except Exception as e:
        print("Error in vector search:", e)
        return []

def build_prompt(user_q: str, chunks: List[dict]) -> str:
    context_texts = []
    for chunk in chunks:
        doc = chunk["document"]
        url = doc.get("url", "unknown")
        text = doc.get("text", "")
        snippet = text[:160].replace("\n", " ").strip()
        context_texts.append(f"URL: {url}\nText: {snippet}")

    context = "\n\n---\n\n".join(context_texts)

    example_json_template = """
{
  "answer": "<your answer here>",
  "links": [
    {
      "url": "<url1>",
      "text": "<link text1>"
    },
    {
      "url": "<url2>",
      "text": "<link text2>"
    }
  ]
}
"""

    prompt = f"""
You are a helpful educational assistant for data-science learners.

Use ONLY the context below to answer the question. Your answer should be concise, informative, and strictly based on the context.

Include relevant links from the context as a list of objects with "url" and "text" fields.

Context:
{context}

Question:
{user_q}

Respond ONLY with a JSON object EXACTLY like this structure:

{example_json_template}

Generate your own answer and links based on the context. Do NOT copy the example literally. Do NOT add any text outside this JSON.
"""
    return prompt.strip()


def ask_llm(prompt: str) -> str:
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 500,
    }
    response = requests.post(OLLAMA_URL, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import json
import re

app = FastAPI()

def parse_llm_response(raw_text: str):
    """
    Extract and parse JSON from LLM output wrapped in ```json ... ``` markdown block.
    If parsing fails, return fallback with raw text as answer and empty links.
    """
    # Use regex to extract JSON inside ```json ... ```
    pattern = r"```json\s*(\{.*?\})\s*```"
    match = re.search(pattern, raw_text, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print("JSON decode error:", e)
    # fallback
    return {"answer": raw_text.strip(), "links": []}

@app.post("/api/")
async def answer_question(request: Request):
    data = await request.json()
    question = data.get("question", "")
    if not question:
        return JSONResponse(content={"answer": "Please provide a question.", "links": []})

    try:
        # Step 1: Embed question
        query_embedding = embed_with_nomic(question)

        # Step 2: Search Typesense with embedding
        hits = search_typesense_vector(query_embedding, top_k=TOP_K)

        # Step 3: Build prompt for LLM with retrieved chunks
        prompt = build_prompt(question, hits)

        # Step 4: Ask LLM with prompt
        raw_llm_output = ask_llm(prompt)

        # Step 5: Parse LLM response JSON
        response_json = parse_llm_response(raw_llm_output)

        return JSONResponse(content=response_json)

    except Exception as e:
        print("Error processing question:", e)
        return JSONResponse(content={"answer": "Sorry, something went wrong.", "links": []})