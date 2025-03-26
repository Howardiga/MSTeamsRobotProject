"""
Module for interacting with the Ollama API to query the Deepseek R1 model
"""
import requests
from typing import Dict, Any, List, Optional

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "deepseek-r1:32b"):
        """
        Initialize the Ollama client
        
        Args:
            base_url: Base URL for the Ollama API
            model: Name of the model to use
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_url = f"{self.base_url}/api/generate"
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a response from the LLM
        
        Args:
            prompt: User prompt/question
            system_prompt: Optional system prompt to guide model behavior
            
        Returns:
            Generated response as a string
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            
            return response.json().get("response", "")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to Ollama API: {str(e)}")
    
    def generate_rag_response(self, 
                            query: str, 
                            context_chunks: List[Dict[str, Any]]) -> str:
        """
        Generate a response using RAG (Retrieval-Augmented Generation)
        
        Args:
            query: User query/question
            context_chunks: List of relevant context chunks with their scores
            
        Returns:
            Generated response as a string
        """
        if not context_chunks:
            return "Sorry, I couldn't find relevant information in the knowledge base."
        
        # Build the context string from retrieved chunks
        context = "\n\n".join([chunk["text"] for chunk in context_chunks])
        
        # Create the RAG prompt with improved instructions
        system_prompt = """You are a helpful assistant that provides complete and comprehensive information about Microsoft Teams based on the context provided.

Instructions:
1. Provide only factual information from the provided context.
2. Include ALL steps and details from the context - do not skip or summarize any information.
3. Present information in a well-structured, easy-to-read format using numbered lists for steps when applicable.
4. Do not include any reasoning, thinking process or personal interpretations.
5. If the context doesn't contain the answer, say only 'Sorry, I couldn't find relevant information in the knowledge base.'
6. Do not mention the "context" in your response.
7. Do not include phrases like "based on the context" or "according to the information provided".
8. Keep your answers direct and focused on the question."""
        
        rag_prompt = f"""Context information:
{context}

User Question: {query}
Your Answer (include ALL relevant information from the context, formatted clearly):"""
        
        # Generate response
        return self.generate(rag_prompt, system_prompt) 