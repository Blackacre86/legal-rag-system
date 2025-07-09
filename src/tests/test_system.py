import os
import time
from production_legal_rag import ProductionLegalRAG

# Check if PDF exists
pdf_path = "scheft_2025_manual.pdf"
if not os.path.exists(pdf_path):
    print(f"❌ PDF not found: {pdf_path}")
    print(f"Files in folder: {os.listdir('.')}")
    exit()

print(f"🚀 Testing optimized legal RAG system with {pdf_path}")

# Initialize the optimized system
rag = ProductionLegalRAG()

# Process the document
print("📖 Processing document...")
chunks = rag.process_document(pdf_path, "scheft_manual")
print(f"✅ Created {len(chunks)} legal chunks")

# Build the index
print("🔍 Building search index...")
rag.build_index(chunks)
print("✅ Index built successfully")

# Test some queries
test_queries = [
    "What are the elements of OUI in Massachusetts?",
    "When can police conduct a protective sweep?",
    "What constitutes a valid Miranda waiver?"
]

print("\n🧪 Testing queries:")
print("=" * 50)

for i, query in enumerate(test_queries, 1):
    print(f"\nTest {i}: {query}")
    
    start_time = time.time()
    result = rag.query(query, top_k=5)
    response_time = time.time() - start_time
    
    print(f"⏱️  Response time: {response_time:.2f} seconds")
    print(f"📚 Sources found: {len(result.get('sources', []))}")
    print(f"📖 Answer preview: {result.get('answer', '')[:200]}...")
    
    if result.get('legal_citations'):
        print(f"⚖️  Legal citations: {result['legal_citations'][:3]}")

print("\n✅ Test complete!")