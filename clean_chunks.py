#!/usr/bin/env python3
"""Clean chunks by removing page numbers and headers/footers."""

import pickle
import re
from tqdm import tqdm

def clean_text(text):
    """Remove common PDF artifacts from text."""
    
    # Remove page numbers (common patterns)
    # Matches: "Page 123", "123", "- 123 -", "Page 123 of 456"
    text = re.sub(r'Page\s+\d+\s*(of\s+\d+)?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-–—]\s*\d+\s*[-–—]\s*$', '', text, flags=re.MULTILINE)
    
    # Remove common headers/footers
    # "Massachusetts Criminal Law 2025", "Chapter X", etc.
    text = re.sub(r'Massachusetts\s+Criminal\s+Law\s+\d{4}', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*Chapter\s+[IVXLCDM]+\s*$', '', text, flags=re.MULTILINE)
    
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
    text = re.sub(r'[ \t]+', ' ', text)     # Single spaces
    
    # Remove common OCR artifacts
    text = re.sub(r'[|]{2,}', '', text)     # Multiple pipe characters
    text = re.sub(r'[-]{4,}', '', text)     # Long dashes
    
    return text.strip()

def main():
    """Clean existing chunks and save new version."""
    
    # Load existing chunks
    print("Loading chunks...")
    with open('text_chunks.pkl', 'rb') as f:
        chunks = pickle.load(f)
    
    print(f"Loaded {len(chunks)} chunks")
    
    # Clean each chunk
    cleaned_chunks = []
    for chunk in tqdm(chunks, desc="Cleaning chunks"):
        if isinstance(chunk, str):
            cleaned = clean_text(chunk)
        elif isinstance(chunk, dict) and 'text' in chunk:
            chunk['text'] = clean_text(chunk['text'])
            cleaned = chunk
        else:
            cleaned = chunk
            
        cleaned_chunks.append(cleaned)
    
    # Save cleaned version
    with open('text_chunks_cleaned.pkl', 'wb') as f:
        pickle.dump(cleaned_chunks, f)
    
    print(f"✅ Saved cleaned chunks to text_chunks_cleaned.pkl")
    
    # Show before/after example
    print("\n📊 Example of cleaning:")
    print("BEFORE:", chunks[0][:200] if chunks else "No chunks")
    print("\nAFTER:", cleaned_chunks[0][:200] if cleaned_chunks else "No cleaned chunks")

if __name__ == "__main__":
    main()