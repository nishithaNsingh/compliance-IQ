from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import json
import faiss
from fastembed import TextEmbedding
from openai import OpenAI
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# ================================
# INIT APP
# ================================
app = FastAPI()

# Add CORS BEFORE mounting static files
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_homepage():
    return FileResponse("static/index.html")

# ================================
# LOAD ENV
# ================================
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

# ================================
# LOAD DATA (ON STARTUP)
# ================================
@app.on_event("startup")
def load_system():
    global embeddings, all_chunks, index, model

    print("🔹 Loading embeddings + chunks...")

    embeddings = np.load("processed/embeddings.npy")

    with open("processed/chunks.json", "r", encoding="utf-8") as f:  # FIX: Added encoding
        all_chunks = json.load(f)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print(f"✅ Loaded {len(all_chunks)} chunks")

    print("🔹 Loading embedding model...")
    model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("✅ Model ready")


# ================================
# REQUEST SCHEMA
# ================================
class QueryRequest(BaseModel):
    query: str


# ================================
# HELPER: get embedding as numpy array
# ================================
def embed(text: str) -> np.ndarray:
    # FIX: Convert generator to list first, then to numpy array
    embeddings_list = list(model.embed([text]))
    return np.array(embeddings_list[0])  # Return the first (and only) embedding


# ================================
# SEARCH
# ================================
def search(query, k=3):
    query_embedding = embed(query)
    # FIX: Reshape to 2D array for FAISS
    query_embedding = query_embedding.reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    return [all_chunks[i] for i in indices[0]]


# ================================
# FILTER
# ================================
def filter_relevant_chunks(results, query, threshold=0.5):
    query_emb = embed(query)

    filtered = []
    for r in results:
        chunk_emb = embed(r["text"])
        # FIX: Properly compute cosine similarity
        score = np.dot(query_emb, chunk_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(chunk_emb))
        if score > threshold:
            filtered.append(r)

    return filtered if filtered else results[:3]


# ================================
# LLM
# ================================
def generate_answer(query, context):
    prompt = f"""
You are a compliance assistant.

Answer ONLY using the context below.
Include citations like (Clause X).

If the answer is not in the context, say:
"Not found in provided documents."

Context:
{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",  # FIX: Changed to a reliable model
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content


# ================================
# HALLUCINATION CHECK
# ================================
def grounding_score(answer, context):
    ans_emb = embed(answer)
    ctx_emb = embed(context)
    # FIX: Proper cosine similarity calculation
    return np.dot(ans_emb, ctx_emb) / (np.linalg.norm(ans_emb) * np.linalg.norm(ctx_emb))


def check_hallucination(answer, context, threshold=0.5):
    score = grounding_score(answer, context)
    if score < threshold:
        return "⚠️ Potential hallucination", score
    else:
        return "✅ Grounded", score


# ================================
# MAIN API
# ================================
@app.post("/ask")
def ask_api(req: QueryRequest):

    results = search(req.query, k=3)
    results = filter_relevant_chunks(results, req.query)
    context = "\n\n".join([r["text"] for r in results])

    if not context.strip():
        return {
            "answer": "Not found in provided documents.",
            "status": "⚠️ No context",
            "score": 0,
            "sources": [],
            "evidence": []
        }

    answer = generate_answer(req.query, context)
    status, score = check_hallucination(answer, context)

    return {
        "answer": answer,
        "status": status,
        "score": float(score),
        "sources": list(set(r.get("source", "Unknown") for r in results)),
        "evidence": [
            {
                "source": r.get("source", "Unknown"),
                "snippet": r["text"][:150] + "..."
            }
            for r in results
        ]
    }
