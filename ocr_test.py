import fitz
import pytesseract
from PIL import Image
import io

# Set Tesseract path (adjust if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Test OCR on first page
pdf_path = "scheft_2025_manual.pdf"
doc = fitz.open(pdf_path)

print("🔍 Testing OCR on first page...")

try:
    # Get first page as image
    page = doc.load_page(0)
    pix = page.get_pixmap()
    img_data = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_data))

    # Try OCR
    text = pytesseract.image_to_string(img)
    print(f"✅ OCR extracted {len(text)} characters")
    print(f"First 200 characters:")
    print(text[:200])
    
    if len(text.strip()) > 50:
        print("🎉 OCR is working! Your PDF can be processed.")
    else:
        print("⚠️ OCR extracted very little text")
        
except Exception as e:
    print(f"❌ OCR failed: {e}")

doc.close()