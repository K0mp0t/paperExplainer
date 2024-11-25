from typing import List

import fitz
import numpy as np
import cv2


def extract_text(pdf_path: str) -> str:
    """Extract text from a PDF file"""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_images(pdf_path: str) -> list[np.ndarray[np.uint8]]:
    """Extract images from a PDF file"""
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        pix = page.get_pixmap()
        img_data = pix.tobytes()
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        images.append(img)
    return images

def extract_tables(pdf_path: str) -> list[list[list[str]]]:
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
    doc = fitz.open(pdf_path)
    tables = []
    for page in doc:
        layout = page.get_text("layout")
        table_layouts = [x for x in layout if isinstance(x, dict) and "type" in x and x["type"] == "table"]
        for table_layout in table_layouts:
            table_data = _extract_table_data(table_layout)
            tables.append(table_data)
    return tables

def process_pdf(self, pdf_path):
    """Process a PDF file and extract text, images, and tables"""
    text = self.extract_text(pdf_path)
    images = self.extract_images(pdf_path)
    tables = self.extract_tables(pdf_path)
    return {
        "text": text,
        "images": images,
        "tables": tables
    }
