import fitz  # PyMuPDF

# Test if PDF can be read
pdf_path = "scheft_2025_manual.pdf"
doc = fitz.open(pdf_path)

print(f"📄 PDF has {len(doc)} pages")

# Extract text from first few pages
text = ""
for page_num in range(min(3, len(doc))):
    page = doc.load_page(page_num)
    page_text = page.get_text()
    text += page_text
    print(f"Page {page_num + 1}: {len(page_text)} characters")

print(f"\nFirst 200 characters:")
print(text[:200])

doc.close()