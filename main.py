from utils.retrieval import get_wikipedia_content, create_faiss_index, retrieve_chunks
from utils.generation import generate_answer, setup_qa_pipeline
from utils.preprocessing import split_text
import wikipedia
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def main():
    
    topic = input("Enter a topic to learn about: ")
    document = get_wikipedia_content(topic)
    
    if not document:
        print("Could not retrieve information.")
        return
    
  
    chunks = split_text(document)
 
    embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    embeddings = embedding_model.encode(chunks)
    index = create_faiss_index(embeddings)
    

    query = input("Ask a question about the topic: ")
    retrieved_chunks = retrieve_chunks(query, embedding_model, index, chunks)
    
    qa_pipeline = setup_qa_pipeline()
    answer = generate_answer(query, retrieved_chunks, qa_pipeline)
    answer = generate_answer(query, retrieved_chunks)
    print(f"Answer: {answer}")
if __name__ == "__main__":
    main()