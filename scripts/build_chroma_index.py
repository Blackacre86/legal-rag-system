#!/usr/bin/env python3
"""Build vector index using Chroma (modern approach)."""

import os
import pickle
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ChromaIndexBuilder:
    def __init__(self, chunks_path, persist_dir="data/chroma"):
        self.chunks_path = chunks_path
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Chroma with OpenAI embeddings
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ.get('OPENAI_API_KEY'),
            model_name="text-embedding-3-small"
        )
        
        # Create Chroma client with persistence
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        
    def load_existing_chunks(self):
        """Load chunks from pickle file."""
        print(f"Loading chunks from {self.chunks_path}")
        
        with open(self.chunks_path, 'rb') as f:
            chunks = pickle.load(f)
        
        print(f"Loaded {len(chunks)} chunks")
        return chunks
    
    def prepare_chunks_for_chroma(self, chunks):
        """Format chunks for Chroma database."""
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(tqdm(chunks, desc="Preparing chunks")):
            # Handle different chunk formats
            if isinstance(chunk, str):
                text = chunk
            elif isinstance(chunk, dict) and 'text' in chunk:
                text = chunk['text']
            else:
                text = str(chunk)
            
            # Skip empty chunks
            if not text.strip():
                continue
                
            documents.append(text)
            metadatas.append({
                "chunk_index": i,
                "source": "law_enforcement_manual",
                "chunk_length": len(text)
            })
            ids.append(f"chunk_{i}")
        
        return documents, metadatas, ids
    
    def build_chroma_index(self, documents, metadatas, ids):
        """Build Chroma vector index."""
        print("Building Chroma index...")
        
        # Delete existing collection if it exists
        try:
            self.client.delete_collection("mass_law_enforcement")
        except:
            pass
        
        # Create new collection
        collection = self.client.create_collection(
            name="mass_law_enforcement",
            embedding_function=self.embedding_function,
            metadata={"description": "Massachusetts Law Enforcement Manual"}
        )
        
        # Add documents in batches
        batch_size = 100
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(documents), batch_size), desc="Adding to Chroma"):
            batch_end = min(i + batch_size, len(documents))
            
            collection.add(
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end],
                ids=ids[i:batch_end]
            )
        
        print(f"Added {len(documents)} documents to Chroma")
        return collection
    
    def save_chunks_as_parquet(self, chunks):
        """Save chunks in Parquet format for reference."""
        print("Saving chunks as Parquet...")
        
        records = []
        for i, chunk in enumerate(chunks):
            if isinstance(chunk, str):
                text = chunk
            elif isinstance(chunk, dict):
                text = chunk.get('text', str(chunk))
            else:
                text = str(chunk)
                
            records.append({
                "chunk_id": f"chunk_{i}",
                "text": text,
                "chunk_index": i
            })
        
        df = pd.DataFrame(records)
        parquet_path = self.persist_dir.parent / "chunks.parquet"
        df.to_parquet(parquet_path, index=False)
        print(f"Saved chunks to {parquet_path}")
        
        return df
    
    def run(self):
        """Execute the pipeline."""
        print("=" * 50)
        print("Building Chroma index from existing chunks")
        print("=" * 50)
        
        # Load chunks
        chunks = self.load_existing_chunks()
        
        # Save as Parquet
        df = self.save_chunks_as_parquet(chunks)
        
        # Prepare for Chroma
        documents, metadatas, ids = self.prepare_chunks_for_chroma(chunks)
        
        print(f"\nPrepared {len(documents)} non-empty chunks for indexing")
        
        # Build Chroma index
        collection = self.build_chroma_index(documents, metadatas, ids)
        
        # Test the index
        print("\n🧪 Testing the index with a sample query...")
        results = collection.query(
            query_texts=["Miranda rights"],
            n_results=3
        )
        
        if results['documents'][0]:
            print("✅ Index is working! Found results for 'Miranda rights'")
            print(f"Top result preview: {results['documents'][0][0][:100]}...")
        
        print(f"\n✅ Chroma index successfully built!")
        print(f"📁 Index saved to: {self.persist_dir}")

def main():
    """Main entry point."""
    # Check for API key
    if not os.environ.get('OPENAI_API_KEY'):
        raise ValueError("Please set your OPENAI_API_KEY in the .env file")
    
    # Find chunks file
    if os.path.exists('text_chunks.pkl'):
        chunks_path = 'text_chunks.pkl'
    elif os.path.exists('rag_output/text_chunks.pkl'):
        chunks_path = 'rag_output/text_chunks.pkl'
    else:
        raise FileNotFoundError("Could not find text_chunks.pkl file")
    
    # Build index
    builder = ChromaIndexBuilder(chunks_path)
    builder.run()

if __name__ == "__main__":
    main()