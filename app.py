import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import streamlit as st
from retriever import retrieve_chunks
from generator import generate_answer

st.set_page_config(page_title="RAG - ML Yearning", page_icon="📚")
st.title("📚 Machine Learning Yearning — Q&A")

# Chat history initialize 
if "messages" not in st.session_state:
    st.session_state.messages = []

# old chat 
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
query = st.chat_input("Type your question...")

if query:
    # User message show and save
    with st.chat_message("user"):
        st.write(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # Answer generate
    with st.chat_message("assistant"):
        with st.spinner("Book searching..."):
            chunks = retrieve_chunks(query)
            answer = generate_answer(query, chunks)
        st.write(answer)

        with st.expander("Source chunks"):
            for i, chunk in enumerate(chunks):
                st.markdown(f"**Chunk {i+1}:**")
                st.write(chunk)
                st.divider()

    # Assistant message save
    st.session_state.messages.append({"role": "assistant", "content": answer})