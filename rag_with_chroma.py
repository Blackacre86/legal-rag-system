#!/usr/bin/env python3
"""RAG system using Chroma and GPT-4 for answers."""

import os
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MassLawRAG:
    """Massachusetts Law Enforcement RAG System with Chroma."""
    
    def __init__(self):
        # Initialize OpenAI
        self.client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        
        # Initialize embedding function
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ.get('OPENAI_API_KEY'),
            model_name="text-embedding-3-small"
        )
        
        # Connect to Chroma
        self.chroma_client = chromadb.PersistentClient(path="data/chroma")
        self.collection = self.chroma_client.get_collection(
            name="mass_law_enforcement",
            embedding_function=self.embedding_function
        )
    
    def search(self, query, n_results=5):
        """Search for relevant chunks."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results
    
    def answer_question(self, question):
        """Get a complete answer using GPT-4."""
        # Search for relevant chunks
        results = self.search(question, n_results=5)
        
        # Combine the relevant chunks
        context = "\n\n".join(results['documents'][0])
        
        # Create prompt for GPT-4
        prompt = f"""You are a Massachusetts law enforcement expert assistant. 
        Use the following context to answer the question accurately and completely.
        If the answer is not in the context, say so clearly.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:"""
        
        # Get answer from GPT-4
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for Massachusetts law enforcement officers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=500
        )
        
        answer = response.choices[0].message.content
        
        # Return answer with sources
        return {
            "question": question,
            "answer": answer,
            "sources": results['documents'][0][:3]  # First 3 sources
        }
    
    def interactive_mode(self):
        """Run interactive Q&A mode."""
        print("🚔 Massachusetts Law Enforcement RAG System")
        print("=" * 50)
        print("Ask any question about Massachusetts criminal law.")
        print("Type 'quit' to exit.\n")
        
        while True:
            question = input("\n❓ Your question: ")
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if question.strip():
                print("\n🔍 Searching and generating answer...")
                
                try:
                    result = self.answer_question(question)
                    
                    print("\n📋 Answer:")
                    print("-" * 40)
                    print(result['answer'])
                    
                    print("\n📚 Sources (first 100 chars):")
                    for i, source in enumerate(result['sources'], 1):
                        preview = source[:100] + "..." if len(source) > 100 else source
                        print(f"{i}. {preview}")
                        
                except Exception as e:
                    print(f"\n❌ Error: {str(e)}")

def main():
    """Run the RAG system."""
    rag = MassLawRAG()
    rag.interactive_mode()

if __name__ == "__main__":
    main()