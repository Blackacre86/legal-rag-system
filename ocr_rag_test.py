import fitz
import pytesseract
from PIL import Image
import io
from sentence_transformers import SentenceTransformer
import time

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_with_ocr(pdf_path, max_pages=10):
    """Extract text using OCR - testing first 10 pages"""
    doc = fitz.open(pdf_path)
    all_text = ""
    
    print(f"📖 Processing {min(max_pages, len(doc))} pages with OCR...")
    
    for page_num in range(min(max_pages, len(doc))):
        print(f"   Processing page {page_num + 1}...")
        
        try:
            # Get page as image
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # OCR the image
            page_text = pytesseract.image_to_string(img)
            all_text += f"\n--- PAGE {page_num + 1} ---\n{page_text}\n"
            
        except Exception as e:
            print(f"   ❌ Error on page {page_num + 1}: {e}")
    
    doc.close()
    return all_text

# Test the OCR extraction
pdf_path = "scheft_2025_manual.pdf"
print("🚀 Testing OCR-based RAG system...")

# Extract text from first 10 pages
text = extract_text_with_ocr(pdf_path, max_pages=10)
print(f"✅ Extracted {len(text)} characters from 10 pages")

# Create chunks
chunks = []
pages = text.split("--- PAGE")
for page in pages:
    if len(page.strip()) > 100:
        # Split page into paragraphs
        paragraphs = [p.strip() for p in page.split('\n\n') if len(p.strip()) > 50]
        chunks.extend(paragraphs)

print(f"✅ Created {len(chunks)} chunks")

# Test embedding and search
if chunks:
    print("🔍 Testing search capability...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create embeddings for first 20 chunks (to save time)
    test_chunks = chunks[:20]
    embeddings = model.encode(test_chunks)
    print(f"✅ Created embeddings: {embeddings.shape}")
    
    # Test a search
    query = "Miranda rights"
    query_embedding = model.encode([query])
    
    # Simple similarity search
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    best_match_idx = similarities.argmax()
    
    print(f"\n🔍 Test Query: '{query}'")
    print(f"🎯 Best match (similarity: {similarities[best_match_idx]:.3f}):")
    print(f"📝 {test_chunks[best_match_idx][:300]}...")
    
    print("\n🎉 OCR-based RAG system is working!")
    
else:
    print("❌ No chunks created - OCR may have failed")