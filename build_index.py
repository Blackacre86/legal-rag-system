#!/usr/bin/env python3
"""Build vector index from PDF for RAG."""

import os
import pandas as pd
import fitz  # PyMuPDF
from pathlib import Path
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

class IndexBuilder:
    def __init__(self, pdf_path, data_dir="data"):
        self.pdf_path = Path(pdf_path)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    def extract_text(self):
        doc = fitz.open(str(self.pdf_path))
        all_text = ""
        for page in tqdm(doc, desc="Extracting PDF text"):
            all_text += page.get_text()
        doc.close()
        
        outpath = self.data_dir / "extracted_text.txt"
        with open(outpath, "w", encoding="utf-8") as f:
            f.write(all_text)
        return all_text
    
    def chunk_text(self, text):
        chunks = self.text_splitter.split_text(text)
        records = []
        for i, chunk in enumerate(chunks):
            records.append({
                "id": f"chunk_{i}",
                "text": chunk,
                "chunk_index": i,
                "source": str(self.pdf_path)
            })
        
        df = pd.DataFrame(records)
        df.to_parquet(self.data_dir / "chunks.parquet", index=False)
        return df
    
    def build_index(self, df):
        texts = df["text"].tolist()
        metadatas = df.drop(columns="text").to_dict(orient="records")
        
        vector_store = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
        vector_store.save_local(str(self.data_dir / "vector_store"))
        print(f"Index saved to: {self.data_dir / 'vector_store'}")
    
    def run(self):
        print("Step 1: Extracting text...")
        text = self.extract_text()
        
        print("Step 2: Splitting text into chunks...")
        df = self.chunk_text(text)
        
        print("Step 3: Embedding and indexing...")
        self.build_index(df)
        
        print("✅ Index built and saved!")

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Please set your OPENAI_API_KEY environment variable.")
    
    builder = IndexBuilder("scheft_2025_manual.pdf")
    builder.run()