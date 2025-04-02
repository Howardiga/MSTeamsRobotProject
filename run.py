"""
Simple script to run the MS Teams chatbot locally
"""
import os
import argparse
import subprocess
import time
from dotenv import load_dotenv
import sys
import shutil

try:
    from advanced_pdf_rag import AdvancedPdfRagPipeline
except ImportError as e:
    print(f"Error importing AdvancedPdfRagPipeline: {e}")
    print("Make sure all dependencies are installed:")
    print("pip install flask python-dotenv requests PyPDF2 langchain-text-splitters")
    print("pip install sentence-transformers faiss-cpu numpy langchain-community langchain-ollama")
    print("pip install pytesseract pymupdf camelot-py tabula-py pdf2image")
    sys.exit(1)

def check_tesseract():
    """Check if Tesseract OCR is installed and accessible"""
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract OCR version {version} found")
        return True
    except Exception as e:
        print(f"⚠️ Tesseract OCR not found or not accessible: {e}")
        print("OCR functionality will be limited. Install Tesseract OCR from:")
        print("https://github.com/UB-Mannheim/tesseract/wiki")
        return False

def process_pdf():
    """Process the PDF and build the vector database"""
    # Load environment variables
    load_dotenv()
    
    # Get configuration
    pdf_path = "Office Solutions IT - Microsoft Teams 101 guide.pdf"
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    vector_db_path = os.getenv("VECTOR_DB_PATH", "./vector_db")
    llm_model = os.getenv("LLM_MODEL", "deepseek-r1:32b")
    chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Check if the PDF exists
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return False
    
    print(f"Processing PDF: {pdf_path}")
    
    # Initialize the pipeline
    pipeline = AdvancedPdfRagPipeline(
        pdf_path=pdf_path,
        ollama_base_url=ollama_url,
        persist_directory=vector_db_path,
        embedding_model=llm_model,
        llm_model=llm_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Process the PDF
    pipeline.process_pdf()
    print(f"PDF processed and vector database created at {vector_db_path}")
    return True

def check_ollama():
    """Check if Ollama is running"""
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    
    try:
        import requests
        response = requests.get(f"{ollama_url}/api/tags")
        
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model.get("name") for model in models]
            
            llm_model = os.getenv("LLM_MODEL", "deepseek-r1:32b")
            
            if any(llm_model in name for name in model_names):
                print(f"✅ Ollama is running with model '{llm_model}' available")
                return True
            else:
                print(f"⚠️ Ollama is running but model '{llm_model}' is not available")
                print(f"Available models: {', '.join(model_names)}")
                print(f"Please run: ollama pull {llm_model}")
                return False
        else:
            print("⚠️ Ollama is running but returned an unexpected response")
            return False
    except Exception as e:
        print(f"❌ Ollama is not running: {e}")
        print("Please start Ollama first with 'ollama serve'")
        return False

def run_app(force_reprocess=False):
    """Run the Flask application"""
    env = os.environ.copy()
    
    # Set force reprocess environment variable if needed
    if force_reprocess:
        env["FORCE_REPROCESS"] = "true"
        print("Setting FORCE_REPROCESS=true - vector database will be rebuilt")
    
    subprocess.run(["python", "app.py"], env=env)

def main():
    """Main function"""
    # Load environment variables
    load_dotenv()
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="MS Teams Chatbot Runner")
    parser.add_argument("--process-only", action="store_true", help="Only process the PDF without starting the server")
    parser.add_argument("--force-process", action="store_true", help="Force processing the PDF even if vector database exists")
    args = parser.parse_args()
    
    # Check dependencies
    check_tesseract()
    
    # Check if Ollama is running
    if not check_ollama():
        return
    
    # Check if vector database exists
    vector_db_path = os.getenv("VECTOR_DB_PATH", "./vector_db")
    db_exists = os.path.exists(vector_db_path)
    
    # Process PDF if needed
    if args.force_process or not db_exists:
        print("Processing PDF...")
        if not process_pdf():
            return
    else:
        print(f"Vector database already exists at {vector_db_path}")
        print("Use --force-process to recreate it")
    
    # Run the app if not in process-only mode
    if not args.process_only:
        print("\nStarting web server...")
        run_app(force_reprocess=args.force_process)

if __name__ == "__main__":
    main() 