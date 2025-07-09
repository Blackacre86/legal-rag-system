import pickle
import os

# Load the pickle file to see what's inside
try:
    # First, let's try the old location
    if os.path.exists('text_chunks.pkl'):
        with open('text_chunks.pkl', 'rb') as f:
            chunks = pickle.load(f)
    # Or maybe it's in rag_output folder
    elif os.path.exists('rag_output/text_chunks.pkl'):
        with open('rag_output/text_chunks.pkl', 'rb') as f:
            chunks = pickle.load(f)
    else:
        print("Could not find text_chunks.pkl")
        exit()
    
    # Let's see what we have
    print(f"Found {len(chunks)} chunks")
    print(f"Type of data: {type(chunks)}")
    
    # If it's a list, show the first chunk
    if isinstance(chunks, list) and len(chunks) > 0:
        print(f"\nFirst chunk preview:")
        if isinstance(chunks[0], dict):
            # It might be a dictionary with text
            print(f"Keys in chunk: {chunks[0].keys()}")
            # Try to find the text
            for key in ['text', 'content', 'page_content']:
                if key in chunks[0]:
                    print(f"\nText preview: {chunks[0][key][:200]}...")
                    break
        elif isinstance(chunks[0], str):
            # It might just be strings
            print(f"Text preview: {chunks[0][:200]}...")
        else:
            print(f"Chunk is of type: {type(chunks[0])}")
            
except Exception as e:
    print(f"Error loading pickle file: {e}")