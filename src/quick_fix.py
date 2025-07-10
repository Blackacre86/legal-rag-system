#!/usr/bin/env python3
"""
Re‑embed text in an already‑extracted corpus.
Intended for emergency reindexing when OCR / chunking logic changes.
"""
import pickle, os, sys, pathlib
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()
DATA_DIR = pathlib.Path("data/chroma")
RAW_CHUNKS = pathlib.Path("rag_output/chunks_fixed.pkl")

if not RAW_CHUNKS.exists():
    sys.exit("❌  chunks_fixed.pkl missing – run extract_chunks.py first")

chunks = pickle.loads(RAW_CHUNKS.read_bytes())
texts = [c["content"] for c in chunks]
metas = [{"page": c["page"]} for c in chunks]

client = chromadb.PersistentClient(str(DATA_DIR))
collection = client.get_or_create_collection(
    name="mass_law_enforcement",
    embedding_function=OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        model_name="text-embedding-3-small",
    ),
)
collection.delete()                 # wipe & rebuild
collection.add(documents=texts, metadatas=metas, ids=[f"id_{i}" for i in range(len(texts))])
print(f"✅  Re‑embedded {len(texts)} chunks into {DATA_DIR}")
