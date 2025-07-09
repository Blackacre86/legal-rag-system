import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle
import sys
import traceback

# --- Configuration ---
PDF_PATH = "scheft_2025_manual.pdf"
INDEX_PATH = "legal_manual.faiss"
CHUNKS_PATH = "text_chunks.pkl"
MODEL_NAME = 'all-MiniLM-L6-v2'

# Try to find Tesseract
def find_tesseract():
    possible_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe'.format(os.environ.get('USERNAME', '')),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # Try to run tesseract directly
    try:
        import subprocess
        subprocess.run(['tesseract', '--version'], capture_output=True, check=True)
        return 'tesseract'  # It's in PATH
    except:
        return None

# Set Tesseract path
tesseract_path = find_tesseract()
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    print(f"Found Tesseract at: {tesseract_path}")
else:
    print("ERROR: Tesseract not found! Please install it from:")
    print("https://github.com/UB-Mannheim/tesseract/wiki")
    sys.exit(1)

# --- Dependency Check ---
def check_dependencies():
    required_modules = {
        'fitz': 'PyMuPDF',
        'pytesseract': 'pytesseract',
        'PIL': 'pillow',
        'sentence_transformers': 'sentence-transformers',
        'faiss': 'faiss-cpu or faiss-gpu',
        'numpy': 'numpy'
    }
    
    missing = []
    for module, package in required_modules.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("ERROR: Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall with: pip install " + " ".join(missing))
        sys.exit(1)

# --- Step 1 (OCR) & 2: Ingestion and Chunking ---
def get_chunks_from_scanned_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in current directory: {os.listdir('.')}")
        return []

    try:
        print(f"Opening PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        print(f"PDF has {len(doc)} pages")
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return []

    full_text = ""
    for page_num in range(len(doc)):
        try:
            page = doc.load_page(page_num)
            
            # Try to extract text first (in case it's not a scanned PDF)
            text = page.get_text()
            if text.strip():
                full_text += text + "\n"
                print(f"  - Extracted text from page {page_num + 1}/{len(doc)}")
            else:
                # Fall back to OCR
                pix = page.get_pixmap(dpi=300)
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
                text = pytesseract.image_to_string(image)
                full_text += text + "\n"
                print(f"  - OCR completed on page {page_num + 1}/{len(doc)}")
        except Exception as e:
            print(f"  - Error on page {page_num + 1}: {e}")
            traceback.print_exc()

    print("\nChunking text...")
    # Improved chunking
    chunks = []
    
    # Split by double newlines first
    paragraphs = full_text.split('\n\n')
    
    current_chunk = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # If adding this paragraph would make chunk too long, save current chunk
        if current_chunk and len(current_chunk) + len(para) > 1000:
            if len(current_chunk) > 100:
                chunks.append(current_chunk)
            current_chunk = para
        else:
            current_chunk += "\n\n" + para if current_chunk else para
    
    # Don't forget the last chunk
    if len(current_chunk) > 100:
        chunks.append(current_chunk)
    
    print(f"Created {len(chunks)} chunks.")
    print(f"Average chunk length: {sum(len(c) for c in chunks) / len(chunks) if chunks else 0:.0f} characters")
    
    return chunks

# --- Main Application Logic ---
def main():
    try:
        print("Legal RAG System Starting...")
        print("-" * 50)
        
        # Check dependencies
        check_dependencies()
        
        # Check for existing data
        if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
            print("Pre-processed data not found. Starting full processing pipeline...")
            
            # Load the embedding model
            print("\nLoading sentence transformer model (first time may download ~90MB)...")
            try:
                model = SentenceTransformer(MODEL_NAME)
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Trying alternative model...")
                model = SentenceTransformer('all-mpnet-base-v2')
            
            # Get chunks
            chunks = get_chunks_from_scanned_pdf(PDF_PATH)
            if not chunks:
                print("No text could be extracted. Exiting.")
                return

            # Get embeddings
            print("\nCreating embeddings...")
            embeddings = model.encode(chunks, show_progress_bar=True)
            
            # Create and save index
            create_and_save_data(chunks, embeddings)
        else:
            print("Found existing data. Loading...")

        # Load saved data
        print("\nLoading model and data...")
        model = SentenceTransformer(MODEL_NAME)
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, 'rb') as f:
            chunks = pickle.load(f)
        
        print("\n" + "="*50)
        print("Legal RAG Q&A System Ready!")
        print("="*50)
        print("Ask questions about the manual or type 'quit' to exit.")

        # Q&A Loop
        while True:
            query = input("\nYour Question: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            # Encode and search
            query_embedding = model.encode([query])
            k = min(3, len(chunks))  # Don't try to get more results than chunks
            distances, indices = index.search(np.array(query_embedding).astype('float32'), k)
            
            print("\n--- Top Results ---")
            for i, idx in enumerate(indices[0]):
                if idx < len(chunks):
                    print(f"\n[Result {i+1} - Relevance: {1/(1+distances[0][i]):.2%}]")
                    print(chunks[idx][:500] + "..." if len(chunks[idx]) > 500 else chunks[idx])
                    print("-" * 30)

    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()

# Helper function kept the same
def create_and_save_data(chunks, embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    print(f"Saving FAISS index to {INDEX_PATH}...")
    faiss.write_index(index, INDEX_PATH)
    
    print(f"Saving chunks to {CHUNKS_PATH}...")
    with open(CHUNKS_PATH, 'wb') as f:
        pickle.dump(chunks, f)

if __name__ == "__main__":
    main()