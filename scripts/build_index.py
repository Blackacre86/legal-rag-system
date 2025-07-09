#!/usr/bin/env python3
"""Build vector index from PDF document."""

from dotenv import load_dotenv
load_dotenv()

import os
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IndexBuilder:
    """Build and manage vector index for RAG system."""
    
    def __init__(self, pdf_path: str, output_dir: str = "data"):
        self.pdf_path = pdf_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.embeddings = OpenAIEmbeddings()
        
    def extract_text_from_pdf(self) -> str:
        """Extract text from PDF file."""
        logger.info(f"Extracting text from {self.pdf_path}")
        
        doc = fitz.open(self.pdf_path)
        text = ""
        
        for page_num in tqdm(range(len(doc)), desc="Extracting pages"):
            page = doc[page_num]
            text += page.get_text()
            
        doc.close()
        
        # Save extracted text
        text_path = self.output_dir / "extracted_text.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text)
            
        logger.info(f"Extracted {len(text)} characters")
        return text
        
    def create_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata."""
        logger.info("Creating text chunks")
        
        # Split text
        texts = self.text_splitter.split_text(text)
        
        # Create chunks with metadata
        chunks = []
        for i, chunk_text in enumerate(texts):
            chunk = {
                'id': f'chunk_{i}',
                'text': chunk_text,
                'metadata': {
                    'source': self.pdf_path,
                    'chunk_index': i,
                    'char_count': len(chunk_text)
                }
            }
            chunks.append(chunk)
            
        # Save chunks
        chunks_path = self.output_dir / "chunks.pkl"
        with open(chunks_path, 'wb') as f:
            pickle.dump(chunks, f)
            
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
        
    def build_vector_store(self, chunks: List[Dict[str, Any]]) -> FAISS:
        """Build FAISS vector store from chunks."""
        logger.info("Building vector store")
        
        # Extract texts and metadatas
        texts = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        
        # Create vector store
        vector_store = FAISS.from_texts(
            texts,
            self.embeddings,
            metadatas=metadatas
        )
        
        # Save vector store
        vector_store.save_local(str(self.output_dir / "vector_store"))
        
        logger.info("Vector store built and saved")
        return vector_store
        
    def build(self):
        """Run the complete index building process."""
        logger.info("Starting index build process")
        
        # Extract text
        text = self.extract_text_from_pdf()
        
        # Create chunks
        chunks = self.create_chunks(text)
        
        # Build vector store
        vector_store = self.build_vector_store(chunks)
        
        logger.info("✅ Index build complete!")
        
        # Print statistics
        print(f"\n📊 Index Statistics:")
        print(f"- Total chunks: {len(chunks)}")
        print(f"- Average chunk size: {sum(len(c['text']) for c in chunks) / len(chunks):.0f} chars")
        print(f"- Vector store saved to: {self.output_dir / 'vector_store'}")

def main():
    """Main entry point."""
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("Please set OPENAI_API_KEY environment variable")
        
    # Build index
    builder = IndexBuilder('scheft_2025_manual.pdf')
    builder.build()

if __name__ == "__main__":
    main()