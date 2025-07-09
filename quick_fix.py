# quick_fix.py
from advanced_ocr_legal_rag import AdvancedOCRLegalRAG
import pickle
import os

print("Applying quick fix...")

# Load the raw text that was saved
with open('rag_output/extracted_text.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Create simple chunks without fancy parsing
chunks = []
pages = text.split("=== PAGE")

for page_content in pages[1:]:  # Skip first empty split
    if len(page_content.strip()) < 50:
        continue
    
    # Extract page number
    page_num = int(page_content.split('===')[0].strip())
    clean_text = page_content.split('===', 1)[1].strip()
    
    # Simple chunking - every 500 words
    words = clean_text.split()
    for i in range(0, len(words), 400):  # 400 words with overlap
        chunk_text = ' '.join(words[i:i+500])
        if len(chunk_text) > 100:
            chunks.append({
                'content': chunk_text,
                'page': page_num,
                'citations': [],
                'legal_terms': [],
                'section_type': None,
                'confidence_score': 1.0
            })

print(f"Created {len(chunks)} chunks")

# Save the fixed chunks
with open('rag_output/chunks_fixed.pkl', 'wb') as f:
    pickle.dump(chunks, f)

print("Fix applied! Now building indices...")

# Continue with the build
rag = AdvancedOCRLegalRAG()
rag.chunks = chunks

# Just build the search index
rag.build_indices()
rag._save_system('rag_output')

print("✅ System repaired and ready!")