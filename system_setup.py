from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from pathlib import Path

import os

load_dotenv()

# New folder (7)/rag_chroma_agent/document/FYP_Blog_Site_Final.pdf

file_path = Path(__file__).parent / "document" / "FYP_Blog_Site_Final.pdf"
cromadb = Path(__file__).parent / "Database"

print("file", file_path)
pdf = PyPDFLoader(str(file_path))

# print("pdf :", bool(pdf))
print("pdf :", bool(file_path))

document = pdf.load()
print("document :", bool(document))

textSplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

Chunks = textSplitter.split_documents(documents=document)


embadding = HuggingFaceEmbeddings(model=os.getenv("EMBADDING_MODEL"))

createDB = Chroma.from_documents(
    documents=Chunks, embedding=embadding, persist_directory=str(cromadb)
)

print("cromadb created now")
