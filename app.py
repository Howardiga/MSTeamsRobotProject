"""
Flask application for serving the RAG chatbot API
"""
import os
import json
from flask import Flask, request, jsonify, send_from_directory
from rag_chatbot import RagChatbot
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize the RAG chatbot
chatbot = RagChatbot(
    pdf_path="Office Solutions IT - Microsoft Teams 101 guide.pdf",
    chunk_size=500,
    chunk_overlap=50,
    embedding_model="all-MiniLM-L6-v2",
    ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
    llm_model=os.getenv("LLM_MODEL", "deepseek-r1:32b")
)

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
        
        # Get response from the chatbot
        response = chatbot.get_response(
            query=question,
            top_k=5,
            similarity_threshold=0.3
        )
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.getenv('PORT', 5000))) 