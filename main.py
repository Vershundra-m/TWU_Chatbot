import os
import sys
from google import genai
import config
from scraper import get_clean_text, chunk_text
from database import VectorStore

# Initialize Gemini
os.environ["GEMINI_API_KEY"] = config.GEMINI_API_KEY
gemini_client = genai.Client()

def generate_answer(query, chunks):
    """Generate answer using Gemini"""
    
    context = "\n\n---\n\n".join([chunk['text'] for chunk in chunks])
    
    prompt = f"""Answer based on the context below. If unsure, say "I don't know."

CONTEXT: {context}
QUESTION: {query}
ANSWER:"""
    
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt)
    
    return response.text

def main():
    print(" TWU Informatics Webpage RAG System")
    print("=" * 30)
    
    # Step 1: Scrape and clean
    clean_text = get_clean_text()
    print(f"   Cleaned {len(clean_text)} characters")
    
    # Step 2: Chunk
    chunks = chunk_text(clean_text)
    print(f"   Created {len(chunks)} chunks")
    
    # Step 3: Upload to Pinecone
    db = VectorStore()
    db.upload_chunks(chunks)
    
    # Step 4: Test queries
    test_questions = [
        "What are the application deadlines?",
        "What are the GPA requirements?"
    ]
    
    for q in test_questions:
        print(f"\n Question: {q}")
        retrieved = db.query(q)
        print(f"   Best score: {retrieved[0]['score']:.3f}")
        answer = generate_answer(q, retrieved)
        print(f" Answer: {answer}")
    
    print("RAG System Complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()
