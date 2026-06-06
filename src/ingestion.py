import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config


def load_and_chunk_pdf(pdf_path: str = config.PDF_PATH):
    # PDF load
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    print(f"Total pages loaded: {len(reader.pages)}")

    # making Chunks 
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.create_documents([full_text])
    print(f"Total chunks created: {len(chunks)}")

    return chunks

if __name__ == "__main__":
    chunks = load_and_chunk_pdf()
    print("\nSample chunk:")
    print(chunks[0].page_content)