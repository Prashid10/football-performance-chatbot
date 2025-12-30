from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

# ---------------- Load document ----------------
def load_document(path):
    """Load text from file"""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

# ---------------- Parse Q&A pairs ----------------
def parse_qa_pairs(text):
    """Extract Question-Answer pairs from football_fitness_recovery.txt"""
    qa_pairs = []
    pattern = re.compile(
        r'Question:\s*(.+?)\s*Answer:\s*(.+?)(?=\s*Question:|$)',
        re.IGNORECASE | re.DOTALL
    )
    matches = pattern.findall(text)
    for question, answer in matches:
        q = question.strip()
        a = answer.strip()
        if q and a:
            qa_pairs.append((q, a))
    print(f"Loaded {len(qa_pairs)} Q&A pairs from knowledge base")
    return qa_pairs

# ---------------- Chunking ----------------
def chunk_text(text, chunk_size=200, overlap=50):
    """Split text into overlapping chunks"""
    chunks = []
    qa_pattern = re.compile(
        r'Question:\s*(.+?)\s*Answer:\s*(.+?)(?=\s*Question:|$)',
        re.IGNORECASE | re.DOTALL
    )
    matches = qa_pattern.findall(text)
    for question, answer in matches:
        chunk = f"Question: {question.strip()}\nAnswer: {answer.strip()}"
        chunks.append(chunk)
    if not chunks:
        paragraphs = [p for p in re.split(r'\n\s*\n', text) if p.strip()]
        current = ""
        for para in paragraphs:
            if len(current) + len(para) > chunk_size and current:
                chunks.append(current)
                current = current[-overlap:]
            current += "\n\n" + para if current else para
        if current:
            chunks.append(current)
    print(f"Created {len(chunks)} chunks")
    return chunks

# ---------------- Build FAISS ----------------
def build_faiss_index(chunks, model_name="all-MiniLM-L6-v2"):
    print(f"Loading embedding model: {model_name}...")
    embedder = SentenceTransformer(model_name)
    print("Encoding chunks...")
    embeddings = embedder.encode(chunks, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"Built FAISS index with {len(chunks)} chunks")
    return index, embeddings, embedder

# ---------------- Retrieve ----------------
def retrieve_chunks(question, chunks, index, embeddings, embedder, top_k=5):
    q_emb = embedder.encode([question])
    distances, indices = index.search(np.array(q_emb), top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        similarity = 1 / (1 + dist)
        results.append((chunks[idx], similarity))
    return results

# ---------------- Extract answer from chunk ----------------
def extract_answer_from_chunk(chunk):
    answer_pattern = re.compile(r'Answer:\s*(.+?)(?=\s*Question:|$)', re.IGNORECASE | re.DOTALL)
    match = answer_pattern.search(chunk)
    if match:
        return match.group(1).strip()
    return chunk.strip()

# ---------------- Create context ----------------
def create_context(retrieved, max_chunks=3):
    context_parts = []
    for chunk, similarity in retrieved[:max_chunks]:
        answer = extract_answer_from_chunk(chunk)
        context_parts.append(answer)
    return " ".join(context_parts)

# ---------------- QA Pipeline ----------------
print("Loading QA model...")
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
print("QA model loaded")

def answer_question(question, context):
    if not context.strip():
        return {
            "answer": "No relevant information found.",
            "confidence": 0.0,
            "context_used": ""
        }
    try:
        result = qa_pipeline(question=question, context=context, handle_impossible_answer=True)
        return {
            "answer": result['answer'],
            "confidence": result['score'],
            "context_used": context[:200] + "..." if len(context) > 200 else context
        }
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "confidence": 0.0,
            "context_used": context[:200]
        }

# ---------------- Exact match ----------------
def exact_match(question, qa_pairs, threshold=0.9):
    from difflib import SequenceMatcher
    question_lower = question.lower().strip().rstrip('?')
    best_match = None
    best_score = 0
    for q, a in qa_pairs:
        q_lower = q.lower().strip().rstrip('?')
        similarity = SequenceMatcher(None, question_lower, q_lower).ratio()
        if similarity > best_score:
            best_score = similarity
            best_match = a
    if best_score >= threshold:
        return best_match, best_score
    return None, best_score

# ---------------- Main ----------------
if __name__ == "__main__":
    doc_path = "data/football_fitness_recovery.txt"
    print("="*70)
    print("FAQ CHATBOT - Loading...")
    print("="*70)
    text = load_document(doc_path)
    qa_pairs = parse_qa_pairs(text)
    chunks = chunk_text(text)
    index, embeddings, embedder = build_faiss_index(chunks)
    print("\n" + "="*70)
    print("FAQ Bot Ready! Type 'quit' to exit.")
    print("="*70 + "\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in ["quit", "exit", "bye"]:
            print("Goodbye!")
            break
        if not q:
            continue

        exact_answer, exact_score = exact_match(q, qa_pairs)
        if exact_answer and exact_score >= 0.9:
            print(f"Bot: {exact_answer}")
            print(f"[Exact Match | Confidence: {exact_score:.2%}]\n")
            continue

        print("Searching knowledge base...")
        retrieved = retrieve_chunks(q, chunks, index, embeddings, embedder, top_k=5)
        print(f"[Top match similarity: {retrieved[0][1]:.2%}]")
        if retrieved[0][1] > 0.8:
            answer = extract_answer_from_chunk(retrieved[0][0])
            print(f"Bot: {answer}")
            print(f"[Direct Match | Confidence: {retrieved[0][1]:.2%}]\n")
        else:
            context = create_context(retrieved, max_chunks=3)
            result = answer_question(q, context)
            print(f"Bot: {result['answer']}")
            print(f"[QA Pipeline | Confidence: {result['confidence']:.2%}]\n")
