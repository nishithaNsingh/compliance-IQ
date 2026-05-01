from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import json
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware

# ================================
# INIT APP
# ================================
app = FastAPI()

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

    with open("processed/chunks.json", "r") as f:
        all_chunks = json.load(f)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print(f"✅ Loaded {len(all_chunks)} chunks")

    print("🔹 Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ Model ready")


# ================================
# REQUEST SCHEMA
# ================================
class QueryRequest(BaseModel):
    query: str


# ================================
# SEARCH
# ================================
def search(query, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)

    return [all_chunks[i] for i in indices[0]]


# ================================
# FILTER (same as notebook)
# ================================
def filter_relevant_chunks(results, query, threshold=0.5):
    query_emb = model.encode([query])

    filtered = []
    for r in results:
        chunk_emb = model.encode([r["text"]])
        score = np.dot(query_emb, chunk_emb.T)[0][0]

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
        model="openai/gpt-oss-120b:free",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content


# ================================
# HALLUCINATION CHECK
# ================================
def grounding_score(answer, context):
    ans_emb = model.encode([answer])
    ctx_emb = model.encode([context])
    return np.dot(ans_emb, ctx_emb.T)[0][0]


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

    # Step 1: search
    results = search(req.query, k=3)

    # Step 2: filter (IMPORTANT — same as notebook)
    results = filter_relevant_chunks(results, req.query)

    # Step 3: context
    context = "\n\n".join([r["text"] for r in results])

    if not context.strip():
        return {
            "answer": "Not found in provided documents.",
            "status": "⚠️ No context",
            "score": 0,
            "sources": [],
            "evidence": []
        }

    # Step 4: LLM
    answer = generate_answer(req.query, context)

    # Step 5: grounding
    status, score = check_hallucination(answer, context)

    # Step 6: response
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
    
    

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for now (dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)