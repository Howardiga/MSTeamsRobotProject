"""
Advanced PDF to RAG ingestion pipeline using open-source tools and Ollama's Deepseek R1:32B model.
This module provides comprehensive PDF processing capabilities including text extraction, OCR,
table extraction, and semantic chunking for optimal RAG performance.
"""
import os
import sys
import tempfile
import warnings
from typing import List, Dict, Any, Optional, Tuple
import fitz  # PyMuPDF
import pytesseract
import camelot
import tabula
import numpy as np
import re
from PIL import Image
import markdown
import pdf2image
from pathlib import Path
import time

# Configure pytesseract path - add this to handle Tesseract not found error
# Common Tesseract installation paths on Windows
TESSERACT_POSSIBLE_PATHS = [
    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
    r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
    r'C:\Tesseract-OCR\tesseract.exe',
    # Add your specific path here if different
]

# Try to set the tesseract command
for path in TESSERACT_POSSIBLE_PATHS:
    if os.path.exists(path):
        pytesseract.pytesseract.tesseract_cmd = path
        print(f"Found Tesseract OCR at: {path}")
        break
    
# LangChain components
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain.schema import Document
from langchain.prompts import PromptTemplate

# Set up warnings
warnings.filterwarnings("ignore", category=UserWarning)

class AdvancedPdfRagPipeline:
    """Advanced PDF to RAG ingestion pipeline with comprehensive extraction capabilities"""
    
    def __init__(self, 
                 pdf_path: str, 
                 ollama_base_url: str = "http://localhost:11434",
                 persist_directory: str = "./vector_db",
                 embedding_model: str = "deepseek-r1:32b",
                 llm_model: str = "deepseek-r1:32b",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the Advanced PDF RAG Pipeline
        
        Args:
            pdf_path: Path to the PDF file
            ollama_base_url: URL for Ollama API
            persist_directory: Directory to store the vector database
            embedding_model: Name of the embedding model to use
            llm_model: Name of the LLM model to use
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between consecutive chunks
        """
        self.pdf_path = pdf_path
        self.ollama_base_url = ollama_base_url
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize embeddings and LLM
        self.embeddings = OllamaEmbeddings(
            base_url=ollama_base_url,
            model=embedding_model
        )
        
        # Initialize vector store if it exists
        self.vector_store = None
        self.all_documents = []  # Store all documents for keyword search
        if os.path.exists(persist_directory):
            try:
                self.vector_store = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embeddings
                )
                print(f"Loaded existing vector store from {persist_directory}")
                
                # Load all documents for keyword search
                try:
                    self.all_documents = self.vector_store.get()["documents"]
                    print(f"Loaded {len(self.all_documents)} documents for hybrid search")
                except Exception as e:
                    print(f"Could not load documents for hybrid search: {str(e)}")
            except Exception as e:
                print(f"Error loading vector store: {str(e)}")
        
        # Initialize the LLM
        self.llm = ChatOllama(
            base_url=ollama_base_url,
            model=llm_model,
            temperature=0.1
        )
    
    def process_pdf(self) -> None:
        """
        Process PDF file through the complete ingestion pipeline
        """
        print(f"Processing PDF: {self.pdf_path}")
        
        # 1. Extract structured text, tables, and images
        markdown_content, metadata = self._extract_pdf_content()
        
        # 2. Chunk the content into semantically coherent passages
        chunks = self._chunk_markdown(markdown_content, metadata)
        
        # 3. Create vector store with embeddings
        self._create_vector_store(chunks)
        
        # 4. Store all documents for keyword search
        self.all_documents = [doc.page_content for doc in chunks]
        print(f"Stored {len(self.all_documents)} documents for hybrid search")
    
    def _extract_pdf_content(self) -> Tuple[str, Dict[str, Any]]:
        """
        Extract structured content from PDF with PyMuPDF, tables with Camelot/Tabula,
        and perform OCR on images with pytesseract
        
        Returns:
            Tuple of (markdown content, metadata dictionary)
        """
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF file not found at {self.pdf_path}")
        
        doc = fitz.open(self.pdf_path)
        
        # PDF metadata
        metadata = {
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
            "page_count": len(doc),
            "source": self.pdf_path
        }
        
        print(f"Extracting content from {metadata['page_count']} pages...")
        
        # Initialize markdown content
        markdown_content = f"# {metadata.get('title') or Path(self.pdf_path).stem}\n\n"
        
        # Process each page
        for page_num, page in enumerate(doc):
            page_markdown = f"## Page {page_num+1}\n\n"
            
            # 1. Extract text
            text = page.get_text()
            if text.strip():
                page_markdown += text + "\n\n"
            
            # 2. Extract tables using Camelot for digital PDFs
            try:
                tables = camelot.read_pdf(self.pdf_path, pages=str(page_num+1))
                if len(tables) > 0:
                    page_markdown += f"### Tables on Page {page_num+1}\n\n"
                    for i, table in enumerate(tables):
                        csv_data = table.df.to_csv(index=False)
                        page_markdown += f"#### Table {i+1}\n\n```csv\n{csv_data}\n```\n\n"
            except Exception as e:
                print(f"Camelot table extraction failed on page {page_num+1}: {str(e)}")
                
                # Fallback to Tabula for scanned PDFs
                try:
                    tables = tabula.read_pdf(self.pdf_path, pages=page_num+1)
                    if len(tables) > 0:
                        page_markdown += f"### Tables on Page {page_num+1}\n\n"
                        for i, table in enumerate(tables):
                            csv_data = table.to_csv(index=False)
                            page_markdown += f"#### Table {i+1}\n\n```csv\n{csv_data}\n```\n\n"
                except Exception as e2:
                    print(f"Tabula table extraction also failed on page {page_num+1}: {str(e2)}")
            
            # 3. Extract and OCR images
            image_list = page.get_images(full=True)
            if image_list:
                page_markdown += f"### Images on Page {page_num+1}\n\n"
                
                for img_index, img_info in enumerate(image_list):
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Save image to temporary file for OCR
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                        temp_file.write(image_bytes)
                        temp_file_path = temp_file.name
                    
                    try:
                        # OCR the image
                        image = Image.open(temp_file_path)
                        try:
                            ocr_text = pytesseract.image_to_string(image)
                            
                            if ocr_text.strip():
                                page_markdown += f"#### Image {img_index+1} OCR Text:\n\n{ocr_text}\n\n"
                        except pytesseract.pytesseract.TesseractNotFoundError:
                            page_markdown += f"#### Image {img_index+1} OCR Text:\n\n[OCR skipped: Tesseract not found]\n\n"
                            print("Warning: Tesseract OCR not available. Install Tesseract and make sure it's in your PATH or set pytesseract.pytesseract.tesseract_cmd manually.")
                    except Exception as e:
                        print(f"OCR failed for image {img_index+1} on page {page_num+1}: {str(e)}")
                    finally:
                        # Clean up temporary file
                        if os.path.exists(temp_file_path):
                            os.unlink(temp_file_path)
            
            # Add page content to main markdown
            markdown_content += page_markdown
        
        # Close the document
        doc.close()
        
        return markdown_content, metadata
    
    def _chunk_markdown(self, markdown_content: str, metadata: Dict[str, Any]) -> List[Document]:
        """
        Chunk markdown content into semantically coherent passages
        
        Args:
            markdown_content: Markdown content to chunk
            metadata: Metadata to include with each chunk
            
        Returns:
            List of Document objects
        """
        print(f"Chunking content with size={self.chunk_size}, overlap={self.chunk_overlap}")
        
        # First, create a more semantic splitting approach
        # Split on headers, but with larger overlap to capture cross-page information
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n## Page", "\n## ", "\n### ", "\n#### ", "\n", " ", ""],
            length_function=len,
            is_separator_regex=False
        )
        
        # Create documents with metadata
        documents = []
        chunks = text_splitter.split_text(markdown_content)
        
        for i, chunk in enumerate(chunks):
            # Include page numbers in chunk metadata when possible
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_id"] = i
            
            # Try to extract all page numbers from chunk to facilitate cross-page retrieval
            page_matches = re.findall(r"Page (\d+)", chunk)
            if page_matches:
                # Store primary page (first mentioned)
                chunk_metadata["page"] = int(page_matches[0])
                # Store all pages mentioned in the chunk
                chunk_metadata["all_pages"] = [int(p) for p in page_matches]
            
            doc = Document(page_content=chunk, metadata=chunk_metadata)
            documents.append(doc)
        
        print(f"Created {len(documents)} chunks from PDF content")
        return documents
    
    def _create_vector_store(self, documents: List[Document]) -> None:
        """
        Create and persist vector store from documents
        
        Args:
            documents: List of Document objects
        """
        print(f"Creating vector store with {len(documents)} documents")
        
        # Create directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Create and persist vector store
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        # Persist to disk
        self.vector_store.persist()
        print(f"Vector store created and persisted to {self.persist_directory}")
    
    def _keyword_search(self, query: str, documents: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Perform a simple keyword search on the documents
        
        Args:
            query: Search query
            documents: List of document texts
            top_k: Number of top results to return
            
        Returns:
            List of (document, score) tuples
        """
        # Split query into keywords
        keywords = [k.lower() for k in query.split() if len(k) > 3]
        
        # Score documents based on keyword matches
        scored_docs = []
        for doc in documents:
            doc_lower = doc.lower()
            score = sum(1 for k in keywords if k in doc_lower)
            
            # Boost score for exact phrase match
            if query.lower() in doc_lower:
                score += 5
                
            # Add specific weights for buttons, menus, etc.
            ui_terms = ["button", "click", "menu", "select", "create", "new", "add", "icon", "plus"]
            for term in ui_terms:
                if term in doc_lower:
                    score += 0.5
            
            scored_docs.append((doc, score))
        
        # Sort by score and return top_k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [(doc, score) for doc, score in scored_docs[:top_k] if score > 0]
    
    def hybrid_search(self, query: str, semantic_top_k: int = 8, keyword_top_k: int = 5) -> List[Document]:
        """
        Perform a hybrid search using both semantic and keyword search
        
        Args:
            query: Search query
            semantic_top_k: Number of top results to retrieve with semantic search
            keyword_top_k: Number of top results to retrieve with keyword search
            
        Returns:
            List of Document objects
        """
        # 1. Semantic search
        semantic_docs = self.vector_store.similarity_search(query, k=semantic_top_k)
        semantic_texts = [doc.page_content for doc in semantic_docs]
        
        # 2. Keyword search (if documents available)
        keyword_results = []
        if self.all_documents:
            keyword_results = self._keyword_search(query, self.all_documents, keyword_top_k)
        
        # 3. Combine results, removing duplicates from keyword search
        combined_docs = semantic_docs.copy()
        
        for doc_text, score in keyword_results:
            # Only add if not already in semantic results
            if doc_text not in semantic_texts:
                # Find the document in the collection
                for i, metadata in enumerate(self.vector_store.get()["metadatas"]):
                    if i < len(self.all_documents) and self.all_documents[i] == doc_text:
                        combined_docs.append(Document(
                            page_content=doc_text,
                            metadata=metadata
                        ))
                        break
        
        return combined_docs
    
    def query(self, user_query: str, top_k: int = 12) -> Dict[str, Any]:
        """
        Query the RAG system with a user question
        
        Args:
            user_query: User question or query
            top_k: Number of top chunks to retrieve (increased from 8 to 12 to capture cross-page info)
            
        Returns:
            Dictionary containing the answer and sources
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Process a PDF first or load an existing one.")
        
        print(f"Query: '{user_query}'")
        
        # 1. Use hybrid search to retrieve relevant chunks
        retrieval_start = time.time()
        
        # Use hybrid search if possible, fall back to semantic search
        try:
            if self.all_documents:
                retrieved_docs = self.hybrid_search(user_query, semantic_top_k=top_k, keyword_top_k=8)
                print(f"Using hybrid search: retrieved {len(retrieved_docs)} documents")
            else:
                retrieved_docs = self.vector_store.similarity_search(user_query, k=top_k)
                print(f"Using semantic search: retrieved {len(retrieved_docs)} documents")
        except Exception as e:
            print(f"Error in hybrid search: {str(e)}. Falling back to semantic search.")
            retrieved_docs = self.vector_store.similarity_search(user_query, k=top_k)
            
        retrieval_time = time.time() - retrieval_start
        
        if not retrieved_docs:
            return {
                "answer": "No relevant information found in the document.",
                "sources": [],
                "performance": {"retrieval_time": retrieval_time}
            }
        
        # 2. Format sources for citation
        sources = []
        context_chunks = []
        
        # Group documents by page for better visualization of multi-page information
        page_docs = {}
        for doc in retrieved_docs:
            page = doc.metadata.get("page", "Unknown")
            if page not in page_docs:
                page_docs[page] = []
            page_docs[page].append(doc)
        
        # Track which pages have information for this query
        involved_pages = list(page_docs.keys())
        involved_pages.sort()
        
        # Process all documents, sorted by page number
        i = 1  # Citation counter
        for page in involved_pages:
            for doc in page_docs[page]:
                source = {
                    "chunk_id": doc.metadata.get("chunk_id", i-1),
                    "page": page,
                    "all_pages": doc.metadata.get("all_pages", [page]),
                    "text": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                }
                sources.append(source)
                
                # Add to context with citation marker
                context_chunks.append(f"[{i}] {doc.page_content}")
                i += 1
        
        # 3. Create prompt with citations and multi-page awareness
        prompt_template = """Answer the user question based on the following context information from a Microsoft Teams guide. 
Include specific references to your sources using the citation markers [1], [2], etc.

Be thorough in your examination of the context - the answer may require you to combine information from multiple chunks or
look for implicit instructions ACROSS DIFFERENT PAGES of the document. The context may contain information from multiple pages.
Pay special attention to any UI elements, buttons, menu options, or step-by-step procedures mentioned in the context.

IMPORTANT: This question may require integrating information from multiple pages or from both text and images. 
Make sure to examine ALL provided chunks carefully, even if they're from different pages, and provide a complete answer.

If the context contains ANY information that could help answer the question, even partially, provide that information clearly.
Only say "I don't have enough information to answer this question" if there is absolutely nothing relevant in the context.

CONTEXT:
{context}

USER QUESTION:
{question}

YOUR ANSWER (with citation markers [1], [2], etc. where appropriate, and mention page numbers when referring to specific information):"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # 4. Generate response with LLM
        generation_start = time.time()
        context_text = "\n\n".join(context_chunks)
        final_prompt = prompt.format(context=context_text, question=user_query)
        
        answer = self.llm.invoke(final_prompt).content
        generation_time = time.time() - generation_start
        
        # 5. Return result with performance metrics and multi-page info
        result = {
            "answer": answer,
            "sources": sources,
            "involved_pages": involved_pages,
            "performance": {
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": retrieval_time + generation_time,
            }
        }
        
        return result

def main():
    """Main function for running the PDF RAG pipeline from command line"""
    if len(sys.argv) < 2:
        print("Usage: python advanced_pdf_rag.py <pdf_path> [query]")
        return
    
    pdf_path = sys.argv[1]
    query = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        # Initialize the pipeline
        pipeline = AdvancedPdfRagPipeline(pdf_path=pdf_path)
        
        # Process PDF if needed (if vector store doesn't exist)
        if not os.path.exists(pipeline.persist_directory):
            pipeline.process_pdf()
        
        # Run query if provided
        if query:
            result = pipeline.query(query)
            print("\n" + "="*50)
            print("ANSWER:")
            print(result["answer"])
            print("\n" + "="*50)
            print("SOURCES:")
            for i, source in enumerate(result["sources"]):
                print(f"[{i+1}] Page {source['page']}: {source['text']}")
            print("\n" + "="*50)
            print(f"Retrieval time: {result['performance']['retrieval_time']:.2f}s")
            print(f"Generation time: {result['performance']['generation_time']:.2f}s")
            print(f"Total time: {result['performance']['total_time']:.2f}s")
    
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 