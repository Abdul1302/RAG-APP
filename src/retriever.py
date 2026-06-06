import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cohere
from pinecone import Pinecone
import config

co = cohere.Client(api_key=config.COHERE_API_KEY)

def embed_query(query: str) -> list[float]:
    response = co.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query"
    )
    return response.embeddings[0]

def retrieve_chunks(query: str) -> list[str]:
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    index = pc.Index(config.PINECONE_INDEX_NAME)

    query_embedding = embed_query(query)

    results = index.query(
        vector=query_embedding,
        top_k=config.TOP_K,
        include_metadata=True
    )

    chunks = [match["metadata"]["text"] for match in results["matches"]]
    return chunks

if __name__ == "__main__":
    query = "What is train/dev/test split?"
    print(f"Query: {query}\n")
    chunks = retrieve_chunks(query)
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} ---")
        print(chunk[:300])
        print()