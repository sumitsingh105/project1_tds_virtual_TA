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


load_dotenv()

# ENV CONFIG
TYPESENSE_API_KEY = os.getenv("TYPESENSE_API_KEY")
TYPESENSE_HOST = os.getenv("TYPESENSE_HOST")
TYPESENSE_PORT = int(os.getenv("TYPESENSE_PORT"))
TYPESENSE_PROTOCOL = os.getenv("TYPESENSE_PROTOCOL")

NOMIC_API_KEY = os.getenv("NOMIC_API_KEY")
NOMIC_URL = "https://api-atlas.nomic.ai/v1/embedding/text"
NOMIC_MODEL = "nomic-embed-text-v1.5"


COLLECTION_NAME = "tds_index_embed_chunks"
TOP_K = 5
EMBED_DIM = 768

# FASTAPI INIT
app = FastAPI(title="Hybrid RAG + AIpipe API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# TYPESENSE INIT
import os
import typesense

typesense_client = typesense.Client({
    "nodes": [{
        "host": os.getenv("TYPESENSE_HOST"),      # your-instance.a1.typesense.net
        "port": int(os.getenv("TYPESENSE_PORT")), # usually 443
        "protocol": os.getenv("TYPESENSE_PROTOCOL")  # "https"
    }],
    "api_key": os.getenv("TYPESENSE_API_KEY"),
    "connection_timeout_seconds": 10
})



AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
AIPIPE_URL = "https://aipipe.org/openrouter/v1/chat/completions"
AIPIPE_MODEL = "google/gemini-2.0-flash-lite-001"


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
    return response["results"][0]["hits"]







# --- PROMPT BUILDER ---
def build_prompt(user_q: str, chunks: List[dict]) -> str:
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
                "text": "My understanding is that you just have to use a tokenizer, similar to what Prof. Anand used, to get the number of tokens and multiply that by the given rate."
            }
        ]
    }

    return f"""
You are a helpful educational assistant for data-science learners.

Use ONLY the context below to answer the question. Provide a JSON response with an "answer" field and a "links" array containing exactly 2 helpful references (url + short summary).

Context:
{context}

Question:
{user_q}

Respond ONLY with a JSON object EXACTLY in this format:
{json.dumps(sample_response, indent=2)}

Do NOT add markdown, explanations, or extra formatting. Output only the JSON object.
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


# --- GEMINI IMAGE DESCRIPTION ---
import base64
from PIL import Image
from io import BytesIO

def describe_image_with_gemini(image_bytes: bytes) -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash")

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    prompt = "Describe this image for a student in one short paragraph. Skip any irrelevant visual features."

    try:
        response = model.generate_content([prompt, image])
        return response.text.strip()
    except Exception as e:
        print("❌ Gemini image error:", e)
        return ""

# --- MAIN API ---


from fastapi import UploadFile, File, Form, Request

@app.post("/api/")
async def rag_answer(
    request: Request,
    question: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    try:
        # Handle JSON
        if request.headers.get("content-type", "").startswith("application/json"):
            data = await request.json()
            question = data.get("question", question)

        if not question and not image:
            raise HTTPException(status_code=400, detail="No question or image provided.")

        # --- If image is provided, use Gemini to describe it ---
        if image:
            image_bytes = await image.read()
            image_description = describe_image_with_gemini(image_bytes)
            if not image_description:
                raise HTTPException(status_code=500, detail="Gemini failed to describe image.")
            question = image_description  # Replace question with caption
            print("🖼️ Gemini caption:", question)

        # Embed and search
        embedding = embed_with_nomic(question)
        chunks = search_typesense_vector(embedding, top_k=TOP_K)
        prompt = build_prompt(question, chunks)

        # Call AI Pipe
        response = requests.post(
            AIPIPE_URL,
            headers={
                "Authorization": f"Bearer {AIPIPE_TOKEN}",
                "Content-Type": "application/json",
            },
            json={
                "model": AIPIPE_MODEL,
                "messages": [{"role": "user", "content": prompt}]
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
    uvicorn.run("app:app", host="0.0.0.0", port=10000)



