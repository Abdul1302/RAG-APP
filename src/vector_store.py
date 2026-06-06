import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cohere
from pinecone import Pinecone
import config

co = cohere.Client(api_key=config.COHERE_API_KEY)

def embed_texts(texts: list[str]) -> list[list[float]]:
    response = co.embed(
        texts=texts,
        model="embed-english-v3.0",
        input_type="search_document"
    )
    return response.embeddings

def store_chunks(chunks):
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    index = pc.Index(config.PINECONE_INDEX_NAME)

    texts = [c.page_content for c in chunks]
    print(f"Embedding {len(texts)} chunks...")

    batch_size = 50
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embeddings = embed_texts(batch)
        vectors = [
            {
                "id": f"chunk-{i+j}",
                "values": embeddings[j],
                "metadata": {"text": batch[j]}
            }
            for j in range(len(batch))
        ]
        index.upsert(vectors=vectors)
        print(f"Stored chunks {i} to {i+len(batch)}")

    print("\nAll chunks stored in Pinecone!")

if __name__ == "__main__":
    from ingestion import load_and_chunk_pdf
    chunks = load_and_chunk_pdf()
    store_chunks(chunks)