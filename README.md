# Microsoft Teams RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about Microsoft Teams using information from a PDF guide.

## Features

- Extracts text content from a PDF using the `unstructured` library
- Splits text into semantically meaningful chunks
- Generates vector embeddings using sentence-transformers
- Stores embeddings in a FAISS vector database
- Performs semantic search to find relevant information
- Uses Deepseek R1 model via Ollama for generating responses
- Provides a Flask API with a chat endpoint

## NEW: Advanced PDF RAG Pipeline

The project now includes an advanced PDF processing pipeline with the following capabilities:

- **Enhanced PDF extraction** using PyMuPDF for better structured text extraction
- **OCR for images** using pytesseract to extract text from images in PDFs
- **Table extraction** with Camelot and Tabula for capturing tabular data in CSV format
- **Markdown conversion** for better preservation of document structure
- **Semantic chunking** with optimized parameters (chunk_size=1000, overlap=200)
- **Chroma vector database** for persistent storage of embeddings
- **Source citations** in generated answers with page references
- **Interactive query mode** for easy question-answering

### Using the Advanced Pipeline

```bash
# Process a PDF and query it
python rag_demo.py --pdf "Office Solutions IT - Microsoft Teams 101 guide.pdf" --query "How do I create a team?"

# Interactive mode
python rag_demo.py --pdf "Office Solutions IT - Microsoft Teams 101 guide.pdf"

# Force reprocessing of a PDF that was already processed
python rag_demo.py --pdf "Office Solutions IT - Microsoft Teams 101 guide.pdf" --force-process
```

## Prerequisites

- Python 3.8+
- Ollama installed and running locally
- Deepseek model available in Ollama (run `ollama pull deepseek-r1:32b`)
- Tesseract OCR (for text extraction from images in PDFs)

## Installation

1. Clone this repository:
```
git clone <repository-url>
cd <repository-directory>
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Make sure the PDF file is in the project directory:
```
Office Solutions IT - Microsoft Teams 101 guide.pdf
```

4. Configure the environment variables in `.env` file (if needed):
```
OLLAMA_URL=http://localhost:11434
LLM_MODEL=deepseek-r1:32b
PORT=5000
```

## Running the Application

1. Make sure Ollama is running with the Deepseek model:
```
ollama run deepseek-r1:32b
```

2. Run the Flask application:
```
python app.py
```

3. Access the web interface at:
```
http://localhost:5000
```

## API Usage

The chatbot provides an API endpoint at `/api/ask` that accepts POST requests with the following JSON structure:

```json
{
  "question": "How do I create a team in Microsoft Teams?"
}
```

The response will be in the following format:

```json
{
  "answer": "The generated answer from the chatbot...",
  "source": "pdf",
  "debug": {
    "topMatchScore": 0.85,
    "totalChunks": 120,
    "matchedChunks": 3
  }
}
```

## Project Structure

- `app.py`: Main Flask application
- `rag_chatbot.py`: Core RAG chatbot implementation
- `pdf_extractor.py`: Module for extracting text from PDFs
- `text_processor.py`: Module for text chunking and embeddings generation
- `llm_client.py`: Client for interacting with Ollama API
- `advanced_pdf_rag.py`: **NEW** Advanced PDF processing and RAG pipeline
- `rag_demo.py`: **NEW** Interactive demo for the advanced pipeline
- `requirements.txt`: Python dependencies
- `.env`: Environment variables
- `chat.html`: Frontend interface 

## Advanced Features Comparison

| Feature | Basic Pipeline | Advanced Pipeline |
|---------|---------------|------------------|
| PDF Text Extraction | PyPDF2 (basic) | PyMuPDF (advanced) |
| OCR Support | No | Yes (pytesseract) |
| Table Extraction | No | Yes (Camelot & Tabula) |
| Images Processing | No | Yes |
| Markdown Conversion | No | Yes |
| Default Chunk Size | 500 | 1000 |
| Vector Store | FAISS (in-memory) | Chroma (persistent) |
| Embedding Model | all-MiniLM-L6-v2 | deepseek-r1:32b |
| Source Citations | No | Yes |
| Interactive Mode | No | Yes | 