import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groq import Groq
import config

client = Groq(api_key=config.GROQ_API_KEY)

def generate_answer(query: str, chunks: list[str], chat_history: list = []) -> str:
    context = "\n\n".join(chunks)

    system_prompt = f"""You are a helpful assistant answering questions about 
Andrew Ng's Machine Learning Yearning book. 
Answer based on the context provided. Be detailed and helpful.
If the answer is in the context, explain it clearly.

Context from the book:
{context}"""

    messages = [{"role": "system", "content": system_prompt}]
    
    # Chat history add karo
    for msg in chat_history:
        messages.append(msg)
    
    # Current question add karo
    messages.append({"role": "user", "content": query})

    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=messages,
        temperature=0.3
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    from retriever import retrieve_chunks
    query = "What is machine learning?"
    chunks = retrieve_chunks(query)
    answer = generate_answer(query, chunks)
    print(f"Question: {query}")
    print(f"\nAnswer: {answer}")