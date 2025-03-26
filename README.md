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

## Prerequisites

- Python 3.8+
- Ollama installed and running locally
- Deepseek model available in Ollama (run `ollama pull deepseek:latest`)
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
LLM_MODEL=deepseek:latest
PORT=5000
```

## Running the Application

1. Make sure Ollama is running with the Deepseek model:
```
ollama run deepseek:latest
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
- `requirements.txt`: Python dependencies
- `.env`: Environment variables
- `chat.html`: Frontend interface 