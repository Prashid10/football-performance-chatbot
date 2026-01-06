import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher

# ---------------- Load document ----------------
def load_document(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

# ---------------- Parse Q&A pairs ----------------
def parse_qa_pairs(text):
    qa_pairs = []
    pattern = re.compile(
        r'Question:\s*(.+?)\s*Answer:\s*(.+?)(?=\s*Question:|$)',
        re.IGNORECASE | re.DOTALL
    )
    matches = pattern.findall(text)
    for q, a in matches:
        qa_pairs.append((q.strip(), a.strip()))
    return qa_pairs

# ---------------- Chunking ----------------
def chunk_text(text):
    chunks = []
    pattern = re.compile(
        r'Question:\s*(.+?)\s*Answer:\s*(.+?)(?=\s*Question:|$)',
        re.IGNORECASE | re.DOTALL
    )
    matches = pattern.findall(text)
    for q, a in matches:
        chunks.append(f"Question: {q.strip()}\nAnswer: {a.strip()}")
    return chunks

# ---------------- Build FAISS ----------------
def build_faiss_index(chunks, model_name="all-MiniLM-L6-v2"):
    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(chunks, show_progress_bar=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, embeddings, embedder

# ---------------- Retrieve chunks ----------------
def retrieve_chunks(question, chunks, index, embedder, top_k=5):
    q_emb = embedder.encode([question])
    distances, indices = index.search(np.array(q_emb), top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        similarity = 1 / (1 + dist)
        results.append((chunks[idx], similarity))
    return results

# ---------------- Extract answer ----------------
def extract_answer_from_chunk(chunk):
    match = re.search(r'Answer:\s*(.+)', chunk, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else chunk

# ---------------- Create context ----------------
def create_context(retrieved, max_chunks=3):
    context = []
    for chunk, _ in retrieved[:max_chunks]:
        context.append(extract_answer_from_chunk(chunk))
    return " ".join(context)

# ---------------- Exact match ----------------
def exact_match(question, qa_pairs, threshold=0.9):
    q_norm = question.lower().strip().rstrip('?')
    best_score = 0
    best_answer = None

    for q, a in qa_pairs:
        score = SequenceMatcher(
            None, q_norm, q.lower().strip().rstrip('?')
        ).ratio()
        if score > best_score:
            best_score = score
            best_answer = a

    if best_score >= threshold:
        return best_answer, best_score
    return None, best_score
