"""
Module for text chunking and embeddings generation for RAG system
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict, Tuple, Any

class TextProcessor:
    def __init__(self, 
                 chunk_size: int = 500, 
                 chunk_overlap: int = 50,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the text processor
        
        Args:
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between consecutive chunks
            embedding_model: Name of the sentence-transformers model to use
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize the embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # FAISS index and mapping for retrieval
        self.index = None
        self.chunks = []
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks
        
        Args:
            text: Input text
            
        Returns:
            List of text chunks
        """
        return self.splitter.split_text(text)
    
    def generate_embeddings(self, chunks: List[str]) -> np.ndarray:
        """
        Generate embeddings for text chunks
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Array of embeddings
        """
        return self.embedding_model.encode(chunks, convert_to_numpy=True)
    
    def build_index(self, text: str) -> None:
        """
        Process text, generate chunks and embeddings, and build FAISS index
        
        Args:
            text: Input text
        """
        # Split text into chunks
        self.chunks = self.split_text(text)
        
        if not self.chunks:
            raise ValueError("No chunks were generated from the input text")
        
        # Generate embeddings
        embeddings = self.generate_embeddings(self.chunks)
        
        # Create FAISS index
        vector_dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(vector_dimension)  # Inner product similarity (cosine after normalization)
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to the index
        self.index.add(embeddings)
    
    def search(self, query: str, top_k: int = 3, similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks using a query
        
        Args:
            query: Query string
            top_k: Number of top results to retrieve
            similarity_threshold: Minimum similarity score threshold
            
        Returns:
            List of dictionaries containing chunk text and similarity score
        """
        if self.index is None or not self.chunks:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Generate embedding for the query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # Normalize query vector for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search the index
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= similarity_threshold and idx < len(self.chunks):
                results.append({
                    "text": self.chunks[idx],
                    "score": float(score)
                })
        
        # Print debug information
        scores_str = [f"{score:.2f}" for score in [r["score"] for r in results]]
        print(f"Query: '{query}', Found {len(results)} results with scores: {scores_str}")
        
        return results 