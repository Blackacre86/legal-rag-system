# test_search.py
from advanced_ocr_legal_rag import AdvancedOCRLegalRAG
import os
import sys

def test_system():
    print("Loading RAG system...")
    
    # Check if system was built
    if not os.path.exists("rag_output"):
        print("❌ Error: System not built yet!")
        print("Run: python build_system.py first")
        sys.exit(1)
    
    # Load the system
    try:
        rag = AdvancedOCRLegalRAG()
        rag.load_system("rag_output")
        print("✅ System loaded successfully!")
        
        # Show system stats
        stats = rag._calculate_system_stats()
        print(f"\nSystem contains:")
        print(f"  - {stats['total_chunks']:,} text chunks")
        print(f"  - {stats['total_pages']} pages")
        print(f"  - {stats['unique_citations']:,} unique citations")
        
    except Exception as e:
        print(f"❌ Error loading system: {e}")
        sys.exit(1)
    
    # Test searches
    print("\n" + "="*60)
    print("TESTING SEARCH FUNCTIONALITY")
    print("="*60)
    
    test_queries = [
        "Miranda rights",
        "probable cause for arrest", 
        "search warrant exceptions",
        "OUI penalties"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        try:
            results = rag.search(query, top_k=3, search_type='hybrid')
            
            print(f"Found {results['total_found']} results in {results['search_time']:.3f} seconds")
            
            for i, result in enumerate(results['results'], 1):
                print(f"\n--- Result {i} ---")
                print(f"📄 Page: {result['page']}")
                print(f"📊 Score: {result['score']:.3f}")
                print(f"📝 Content: {result['highlighted_content'][:300]}...")
                
                if result['citations']:
                    print(f"📎 Citations: {', '.join(result['citations'][:3])}")
                if result['legal_terms']:
                    print(f"⚖️ Legal terms: {', '.join(result['legal_terms'][:5])}")
                print(f"🎯 Relevance: {result['relevance_explanation']}")
                
        except Exception as e:
            print(f"❌ Search error: {e}")
    
    # Interactive mode
    print("\n" + "="*60)
    print("INTERACTIVE SEARCH MODE")
    print("Type 'quit' to exit")
    print("="*60)
    
    while True:
        query = input("\nEnter search query: ").strip()
        if query.lower() == 'quit':
            break
            
        if not query:
            continue
            
        try:
            results = rag.search(query, top_k=5)
            print(f"\nFound {results['total_found']} results:")
            
            for i, result in enumerate(results['results'], 1):
                print(f"\n{i}. Page {result['page']} (Score: {result['score']:.3f})")
                print(f"   {result['highlighted_content'][:200]}...")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_system()