# build_system.py
"""
Script to build the complete OCR Legal RAG system
"""

from advanced_ocr_legal_rag import AdvancedOCRLegalRAG
import logging
import sys
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    # Check if PDF path provided
    if len(sys.argv) < 2:
        # If not, look for PDF in current directory
        pdfs = [f for f in os.listdir('.') if f.endswith('.pdf')]
        if pdfs:
            pdf_path = pdfs[0]
            print(f"Found PDF: {pdf_path}")
        else:
            print("No PDF found! Please provide path:")
            print("Usage: python build_system.py <path_to_pdf>")
            sys.exit(1)
    else:
        pdf_path = sys.argv[1]
    
    # Check if PDF exists
    if not os.path.exists(pdf_path):
        print(f"Error: PDF not found at {pdf_path}")
        sys.exit(1)
    
    print("🚀 Building Advanced OCR Legal RAG System")
    print("="*60)
    print(f"PDF: {pdf_path}")
    print(f"PDF Size: {os.path.getsize(pdf_path) / 1024 / 1024:.1f} MB")
    print("This will take 30-60 minutes for a large PDF...")
    print("="*60)
    
    try:
        # Initialize system
        print("\n1. Initializing system...")
        rag = AdvancedOCRLegalRAG(config_path="config.yaml")
        
        # Build the system
        print("\n2. Starting build process...")
        print("   - Extracting text with OCR")
        print("   - Creating legal chunks")
        print("   - Building search indices")
        print("   - Analyzing citations")
        
        result = rag.build_system(
            pdf_path=pdf_path,
            output_dir="rag_output",
            force_rebuild=True
        )
        
        print(f"\n✅ Build completed!")
        print(f"Status: {result['status']}")
        
        if 'stats' in result:
            stats = result['stats']
            print(f"\nSystem Statistics:")
            print(f"  Total chunks: {stats['total_chunks']:,}")
            print(f"  Total pages: {stats['total_pages']:,}")
            print(f"  Unique citations: {stats['unique_citations']:,}")
            print(f"  OCR confidence: {stats.get('ocr_confidence', 0):.1%}")
        
        print(f"\nOutput saved to: {os.path.abspath('rag_output')}")
        
    except Exception as e:
        print(f"\n❌ Error during build: {e}")
        print("\nTroubleshooting:")
        print("1. Check that all files are created (advanced_ocr_legal_rag.py)")
        print("2. Verify all dependencies are installed")
        print("3. Check config.yaml exists")
        raise

if __name__ == "__main__":
    main()