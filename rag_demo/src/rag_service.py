"""
RAG Service - Main interface for Retrieval-Augmented Generation
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_wrapper import LLMWrapper

class RAGService:
    """RAG service for intrusion detection queries"""
    
    def __init__(self, collection_name: str = 'intrusion_detection_data',
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 llm_model: str = "phi3:mini",
                 use_ollama: bool = True,
                 top_k: int = 3):
        """
        Initialize RAG service
        
        Args:
            collection_name: ChromaDB collection name
            embedding_model: Embedding model name
            llm_model: LLM model name
            use_ollama: Whether to use Ollama
            top_k: Number of top chunks to retrieve
        """
        self.top_k = top_k
        
        # Initialize embedding model
        print("Loading embedding model...")
        self.embedder = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB - use same directory as ingestion
        chroma_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'chroma_db')
        
        if not os.path.exists(chroma_dir):
            raise ValueError(f"ChromaDB directory not found: {chroma_dir}\nPlease run data ingestion first: python src/rag_data_ingestion.py")
        
        self.client = chromadb.PersistentClient(path=chroma_dir)
        
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"✓ Loaded collection: {collection_name}")
            print(f"  Collection has {self.collection.count()} chunks")
        except Exception as e:
            raise ValueError(f"Collection {collection_name} not found. Please run data ingestion first: python src/rag_data_ingestion.py\nError: {e}")
        
        # Initialize LLM
        self.llm = LLMWrapper(model_name=llm_model, use_ollama=use_ollama)
    
    def retrieve_context(self, query: str) -> List[Dict]:
        """
        Retrieve relevant context chunks for a query
        
        Args:
            query: User query
            
        Returns:
            List of relevant chunks with metadata
        """
        # Generate query embedding
        query_embedding = self.embedder.encode(query).tolist()
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k
        )
        
        # Format results
        contexts = []
        if results['documents'] and len(results['documents']) > 0 and len(results['documents'][0]) > 0:
            for i, doc in enumerate(results['documents'][0]):
                contexts.append({
                    'text': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] and len(results['metadatas']) > 0 else {},
                    'distance': results['distances'][0][i] if results['distances'] and len(results['distances']) > 0 else None
                })
        
        return contexts
    
    def query(self, user_query: str) -> Dict:
        """
        Process a user query with RAG
        
        Args:
            user_query: User's question
            
        Returns:
            Dictionary with answer, context, and metadata
        """
        # Retrieve relevant context
        contexts = self.retrieve_context(user_query)
        
        # Combine contexts
        context_text = "\n\n---\n\n".join([
            f"Context {i+1} (from {ctx['metadata'].get('filename', 'unknown')}):\n{ctx['text']}"
            for i, ctx in enumerate(contexts)
        ])
        
        # Generate response with LLM
        answer = self.llm.generate(
            prompt=user_query,
            context=context_text if contexts else None
        )
        
        return {
            'answer': answer,
            'contexts': contexts,
            'num_contexts': len(contexts),
            'query': user_query
        }

if __name__ == "__main__":
    # Test RAG service
    print("Testing RAG service...")
    try:
        rag = RAGService()
        result = rag.query("What are the characteristics of DDoS attacks in the dataset?")
        print("\n" + "="*60)
        print("Answer:")
        print("="*60)
        print(result['answer'])
        print(f"\nUsed {result['num_contexts']} context chunks")
    except Exception as e:
        print(f"Error: {e}")

