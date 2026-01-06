from transformers import pipeline
from utils import (
    load_document,
    parse_qa_pairs,
    chunk_text,
    build_faiss_index,
    retrieve_chunks,
    extract_answer_from_chunk,
    create_context,
    exact_match
)

# ---------------- Load QA Model ----------------
print("Loading QA model...")
qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad"
)
print("QA model loaded")

def answer_question(question, context):
    if not context.strip():
        return "No relevant information found.", 0.0

    result = qa_pipeline(
        question=question,
        context=context,
        handle_impossible_answer=True
    )
    return result["answer"], result["score"]

# ---------------- Main ----------------
if __name__ == "__main__":
    print("=" * 70)
    print("FOOTBALL FITNESS & RECOVERY CHATBOT")
    print("=" * 70)

    text = load_document("data/football_fitness_recovery_qa.txt")
    qa_pairs = parse_qa_pairs(text)
    chunks = chunk_text(text)

    index, embeddings, embedder = build_faiss_index(chunks)

    print("Bot is ready! Type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ["quit", "exit", "bye"]:
            print("Bot: Goodbye!")
            break

        exact_answer, score = exact_match(question, qa_pairs)
        if exact_answer:
            print(f"Bot: {exact_answer}")
            print(f"[Exact Match | Confidence: {score:.2%}]\n")
            continue

        retrieved = retrieve_chunks(question, chunks, index, embedder)
        top_chunk, similarity = retrieved[0]

        if similarity > 0.8:
            answer = extract_answer_from_chunk(top_chunk)
            print(f"Bot: {answer}")
            print(f"[Direct Retrieval | Confidence: {similarity:.2%}]\n")
        else:
            context = create_context(retrieved)
            answer, conf = answer_question(question, context)
            print(f"Bot: {answer}")
            print(f"[QA Pipeline | Confidence: {conf:.2%}]\n")

