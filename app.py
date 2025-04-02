"""
Flask application for serving the RAG chatbot API
"""
import os
import json
import shutil
from flask import Flask, request, jsonify, send_from_directory, render_template
from advanced_pdf_rag import AdvancedPdfRagPipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Get configuration from environment
PDF_PATH = "Office Solutions IT - Microsoft Teams 101 guide.pdf"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-r1:32b")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_db")
FORCE_REPROCESS = os.getenv("FORCE_REPROCESS", "false").lower() == "true"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# If force reprocessing is enabled and vector DB exists, delete it
if FORCE_REPROCESS and os.path.exists(VECTOR_DB_PATH):
    print(f"Forcing reprocessing, removing existing vector database at {VECTOR_DB_PATH}")
    shutil.rmtree(VECTOR_DB_PATH)

# Initialize the Advanced PDF RAG Pipeline
print("Using Advanced PDF RAG Pipeline...")
chatbot = AdvancedPdfRagPipeline(
    pdf_path=PDF_PATH,
    ollama_base_url=OLLAMA_URL,
    persist_directory=VECTOR_DB_PATH,
    embedding_model=LLM_MODEL,
    llm_model=LLM_MODEL,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

# Process PDF if vector store doesn't exist yet
if not os.path.exists(VECTOR_DB_PATH):
    print("Building vector database from PDF...")
    chatbot.process_pdf()

@app.route('/')
def index():
    """Serve the index.html file"""
    return send_from_directory('.', 'chat.html')

@app.route('/api/ask', methods=['POST'])
def ask():
    """
    Endpoint for handling chat requests
    
    Expects:
        - JSON with a 'question' field
        
    Returns:
        - JSON with 'answer', 'source', and optional 'debug' fields
    """
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({"error": "Missing 'question' field"}), 400
        
        question = data['question']
        
        # Query the advanced pipeline
        raw_response = chatbot.query(
            user_query=question,
            top_k=12  # Retrieve 12 chunks for better cross-page retrieval
        )
        
        # Format the response
        response = {
            "answer": raw_response["answer"],
            "source": "pdf" if raw_response["sources"] else None,
            "debug": {
                "retrievalTime": raw_response["performance"]["retrieval_time"],
                "generationTime": raw_response["performance"]["generation_time"],
                "matchedChunks": len(raw_response["sources"]),
                "totalChunks": len(chatbot.all_documents) if hasattr(chatbot, 'all_documents') else None,
                "involvedPages": raw_response.get("involved_pages", [])
            },
            "sources": raw_response["sources"]
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/info', methods=['GET'])
def info():
    """Return information about the current configuration"""
    return jsonify({
        "pdfPath": PDF_PATH,
        "usingAdvancedPipeline": True,
        "llmModel": LLM_MODEL,
        "chunking": {
            "size": CHUNK_SIZE,
            "overlap": CHUNK_OVERLAP
        }
    })

@app.route('/api/reprocess', methods=['POST'])
def reprocess():
    """Reprocess the PDF to rebuild the vector database"""
    try:
        # Delete existing vector database
        if os.path.exists(VECTOR_DB_PATH):
            shutil.rmtree(VECTOR_DB_PATH)
        
        # Reprocess the PDF
        chatbot.process_pdf()
        
        return jsonify({
            "success": True,
            "message": "PDF reprocessed successfully",
            "documentCount": len(chatbot.all_documents) if hasattr(chatbot, 'all_documents') else 0
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.getenv('PORT', 5000))) 