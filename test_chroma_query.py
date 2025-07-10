#!/usr/bin/env python3
"""Test querying the Chroma database."""

import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def query_chroma(query_text, num_results=5):
    """Query the Chroma database."""
    
    # Initialize embedding function (same as when building)
    embedding_function = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ.get('OPENAI_API_KEY'),
        model_name="text-embedding-3-small"
    )
    
    # Connect to existing Chroma database
    client = chromadb.PersistentClient(path="data/chroma")
    
    # Get our collection
    collection = client.get_collection(
        name="mass_law_enforcement",
        embedding_function=embedding_function
    )
    
    # Query the collection
    print(f"\n🔍 Searching for: '{query_text}'")
    print("=" * 50)
    
    results = collection.query(
        query_texts=[query_text],
        n_results=num_results
    )
    
    # Display results
    for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
        print(f"\n📄 Result {i+1} (Score: {distance:.4f}):")
        print("-" * 40)
        # Show first 300 characters of each result
        preview = doc[:300] + "..." if len(doc) > 300 else doc
        print(preview)
    
    return results

def main():
    """Run some test queries."""
    
    print("🚔 Massachusetts Law Enforcement RAG - Query Tester")
    print("=" * 50)
    
    # Test queries
    test_queries = [
        "Miranda rights",
        "probable cause for arrest",
        "traffic stop procedures",
        "search warrant requirements",
        "use of force"
    ]
    
    # Run a few automatic tests
    print("\n🧪 Running test queries...")
    query_chroma(test_queries[0], num_results=3)
    
    # Interactive mode
    print("\n\n💬 Interactive Query Mode")
    print("Type 'quit' to exit")
    print("-" * 50)
    
    while True:
        user_query = input("\n❓ Enter your question: ")
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
            
        if user_query.strip():
            query_chroma(user_query, num_results=3)

if __name__ == "__main__":
    main()