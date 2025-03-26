"""
Main module for the RAG chatbot
"""
import os
from typing import Dict, Any, List, Optional

from pdf_extractor import extract_pdf_text
from text_processor import TextProcessor
from llm_client import OllamaClient

class RagChatbot:
    def __init__(self, 
                 pdf_path: str,
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 ollama_url: str = "http://localhost:11434",
                 llm_model: str = "deepseek-r1:32b"):
        """
        Initialize the RAG chatbot
        
        Args:
            pdf_path: Path to the PDF knowledge base
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between consecutive chunks
            embedding_model: Name of the sentence-transformers model
            ollama_url: URL of the Ollama API
            llm_model: Name of the LLM model to use
        """
        self.pdf_path = pdf_path
        
        # Initialize text processor
        self.text_processor = TextProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model
        )
        
        # Initialize LLM client
        self.llm_client = OllamaClient(
            base_url=ollama_url,
            model=llm_model
        )
        
        # Initialize the knowledge base
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """
        Initialize the knowledge base by extracting text from PDF and building the index
        """
        print(f"Extracting text from {self.pdf_path}...")
        text_content = extract_pdf_text(self.pdf_path)
        
        print(f"Building vector index...")
        self.text_processor.build_index(text_content)
        print(f"Knowledge base initialized with {len(self.text_processor.chunks)} chunks")
    
    def get_response(self, 
                    query: str, 
                    top_k: int = 5, 
                    similarity_threshold: float = 0.3) -> Dict[str, Any]:
        """
        Get a response for a user query
        
        Args:
            query: User query/question
            top_k: Number of top context chunks to retrieve
            similarity_threshold: Minimum similarity score threshold
            
        Returns:
            Dictionary containing the response and relevant debug information
        """
        # Retrieve relevant context chunks
        relevant_chunks = self.text_processor.search(
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
        
        # Generate response using RAG
        response = self.llm_client.generate_rag_response(
            query=query,
            context_chunks=relevant_chunks
        )
        
        # Prepare debug information
        debug_info = {
            "topMatchScore": relevant_chunks[0]["score"] if relevant_chunks else None,
            "totalChunks": len(self.text_processor.chunks),
            "matchedChunks": len(relevant_chunks)
        }
        
        result = {
            "answer": response,
            "source": "pdf" if relevant_chunks else None,
            "debug": debug_info
        }
        
        return result 