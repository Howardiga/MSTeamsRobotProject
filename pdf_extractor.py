"""
Module for extracting text content from PDF files using PyPDF2
"""
import os
import PyPDF2
from typing import List, Dict, Any
import warnings

def extract_pdf_text(pdf_path: str) -> str:
    """
    Extract text content from a PDF file using PyPDF2
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text content as a string
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
    
    try:
        text_content = ""
        
        # Open the PDF file
        with open(pdf_path, 'rb') as file:
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Get the number of pages
            num_pages = len(pdf_reader.pages)
            
            print(f"Extracting text from {pdf_path} ({num_pages} pages)...")
            
            # Extract text from each page
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text_content += page.extract_text() + "\n\n"
        
        # Warn if the text content is very short, which might indicate extraction issues
        if len(text_content) < 1000:
            warnings.warn(f"Extracted text is very short ({len(text_content)} characters). This might indicate extraction issues.")
        
        return text_content
    
    except Exception as e:
        # Handle exceptions and provide a more user-friendly error message
        error_msg = f"Error extracting text from PDF: {str(e)}"
        print(error_msg)
        raise RuntimeError(error_msg) 