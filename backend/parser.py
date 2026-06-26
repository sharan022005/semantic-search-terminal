import io
import PIL.Image

# Monkeypatch PIL.Image.ANTIALIAS for compatibility with older libraries (like easyocr) on Pillow 10+
if not hasattr(PIL.Image, "ANTIALIAS"):
    PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS

def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")

_ocr_reader = None

def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        # This will download the models on first invocation if not present
        _ocr_reader = easyocr.Reader(['en'])
    return _ocr_reader

def _run_ocr_on_bytes(image_bytes: bytes) -> str:
    from PIL import Image
    import numpy as np
    
    reader = _get_ocr_reader()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(img)
    
    results = reader.readtext(img_np)
    return " ".join([res[1] for res in results])

def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF is not installed. Please install 'pymupdf'.")
        
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text_content = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text()
        
        # If the page has no native text, try to run OCR
        if not page_text.strip():
            try:
                # Convert page to image bytes
                pix = page.get_pixmap()
                img_data = pix.tobytes("png")
                # Run OCR on image bytes
                page_text = _run_ocr_on_bytes(img_data)
            except ImportError:
                page_text = "[OCR skipped: easyocr or Pillow is not installed]"
            except Exception as e:
                page_text = f"[OCR failed: {e}]"
            
        text_content.append(page_text)
        
    return "\n\n".join(text_content)

def extract_text_from_image(file_bytes: bytes) -> str:
    try:
        return _run_ocr_on_bytes(file_bytes)
    except ImportError:
        raise ImportError("EasyOCR or Pillow is not installed. Please install dependencies.")

def extract_pages(file_bytes: bytes, filename: str) -> list[dict]:
    ext = filename.split(".")[-1].lower()
    if ext == "pdf":
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF is not installed. Please install 'pymupdf'.")
            
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()
            
            # If the page has no native text, try to run OCR
            if not page_text.strip():
                try:
                    pix = page.get_pixmap()
                    img_data = pix.tobytes("png")
                    page_text = _run_ocr_on_bytes(img_data)
                except Exception as e:
                    page_text = f"[OCR failed: {e}]"
            pages.append({"page_num": page_num + 1, "text": page_text})
        return pages
    elif ext in ["png", "jpg", "jpeg", "webp", "bmp"]:
        text = extract_text_from_image(file_bytes)
        return [{"page_num": 1, "text": text}]
    else:
        # Default to txt/plain text
        text = extract_text_from_txt(file_bytes)
        return [{"page_num": 1, "text": text}]

def extract_text(file_bytes: bytes, filename: str) -> str:
    pages = extract_pages(file_bytes, filename)
    return "\n\n".join([p["text"] for p in pages])

