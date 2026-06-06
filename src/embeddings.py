import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pinecone import Pinecone
import config

def get_pinecone_index():
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    index = pc.Index(config.PINECONE_INDEX_NAME)
    return index

if __name__ == "__main__":
    print("Pinecone connection test...")
    index = get_pinecone_index()
    stats = index.describe_index_stats()
    print(f"Index connected successfully!")
    print(f"Stats: {stats}")