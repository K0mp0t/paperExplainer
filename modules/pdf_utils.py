import fitz
import numpy as np
import cv2
from io import IOBase


def extract_text(pdf_path: str = None, stream: IOBase = None) -> str:
    """Extract text from a PDF file"""
    doc = fitz.open(pdf_path, stream, filetype='pdf')
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_images(pdf_path: str = None, stream: IOBase = None) -> list[np.ndarray[np.uint8]]:
    """Extract images from a PDF file"""
    doc = fitz.open(pdf_path, stream, filetype='pdf')
    images = []
    for page in doc:
        for image_data in page.get_images():
            pix = fitz.Pixmap(doc, image_data[0])
            img_data = pix.tobytes()
            img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            images.append(img)
    return images

def extract_tables(pdf_path: str = None, stream: IOBase = None) -> list[list[list[str]]]:
    """Extract tables from a PDF file"""
    def _extract_table_data(table_layout: dict) -> list[list[str]]:
        """Extract table data from a layout"""
        rows = []
        for row in table_layout["cells"]:
            cols = []
            for col in row:
                text = col["text"]
                cols.append(text)
            rows.append(cols)
        return rows
    doc = fitz.open(pdf_path, stream, filetype='pdf')
    tables = []
    for page in doc:
        layout = page.get_text("layout")
        table_layouts = [x for x in layout if isinstance(x, dict) and "type" in x and x["type"] == "table"]
        for table_layout in table_layouts:
            table_data = _extract_table_data(table_layout)
            tables.append(table_data)
    return tables

def process_pdf(pdf_path: str = None, stream: IOBase = None) -> dict:
    """Process a PDF file and extract text, images, and tables"""
    text = extract_text(pdf_path, stream)
    # images = extract_images(pdf_path, stream)
    tables = extract_tables(pdf_path, stream)
    return {
        "text": text,
        # "images": images,
        "tables": tables
    }
