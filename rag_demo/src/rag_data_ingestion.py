"""
RAG Data Ingestion Module
Processes CSV files from data_original/ and Testing_data/ and creates vector embeddings
"""

import os
import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import json
import sys
import warnings

# Suppress pandas warnings about invalid values in statistics
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Add parent directory to path to access CSE543_Group1
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class RAGDataIngestion:
    """Ingest CSV data into vector database for RAG"""
    
    def __init__(self, data_dir: str = None, 
                 testing_data_dir: str = None,
                 collection_name: str = 'intrusion_detection_data',
                 embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        Initialize RAG data ingestion
        
        Args:
            data_dir: Directory containing training CSV files (defaults to CSE543_Group1/data_original)
            testing_data_dir: Directory containing test CSV files (defaults to CSE543_Group1/Testing_data)
            collection_name: Name for ChromaDB collection
            embedding_model: Sentence transformer model name
        """
        # Default to CSE543_Group1 directories if not specified
        if data_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            data_dir = os.path.join(base_dir, 'CSE543_Group1', 'data_original')
        
        if testing_data_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            testing_data_dir = os.path.join(base_dir, 'CSE543_Group1', 'Testing_data')
        
        self.data_dir = data_dir
        self.testing_data_dir = testing_data_dir
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}...")
        self.embedder = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB - store in rag_demo directory
        chroma_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'chroma_db')
        os.makedirs(chroma_dir, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=chroma_dir)
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Intrusion Detection System RAG data"}
            )
            print(f"Created new collection: {collection_name}")
    
    def chunk_csv_data(self, df: pd.DataFrame, filename: str, 
                      chunk_size: int = 100) -> List[Dict]:
        """
        Convert DataFrame rows into text chunks for embedding
        
        Args:
            df: DataFrame to chunk
            filename: Source filename
            chunk_size: Number of rows per chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        chunks = []
        
        for i in range(0, len(df), chunk_size):
            chunk_df = df.iloc[i:i+chunk_size]
            
            # Create descriptive text from chunk
            # Include statistics and sample data
            chunk_text = f"Data from {filename}:\n\n"
            
            # Add summary statistics
            numeric_cols = chunk_df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                chunk_text += "Summary Statistics:\n"
                stats = chunk_df[numeric_cols].describe()
                chunk_text += stats.to_string() + "\n\n"
            
            # Add label distribution if available
            if 'Label' in chunk_df.columns:
                label_dist = chunk_df['Label'].value_counts()
                chunk_text += f"Label Distribution:\n{label_dist.to_string()}\n\n"
            
            # Add sample rows (first 5)
            chunk_text += "Sample Records:\n"
            sample_rows = chunk_df.head(5).to_string()
            chunk_text += sample_rows + "\n"
            
            # Add key features if available
            key_features = ['Flow Duration', 'Total Fwd Packets', 
                          'SYN Flag Count', 'Flow Packets/s', 'Flow Bytes/s']
            available_features = [f for f in key_features if f in chunk_df.columns]
            if available_features:
                chunk_text += f"\nKey Features (mean):\n"
                for feat in available_features:
                    mean_val = chunk_df[feat].mean()
                    chunk_text += f"  {feat}: {mean_val:.2f}\n"
            
            chunks.append({
                'text': chunk_text,
                'metadata': {
                    'filename': filename,
                    'chunk_index': i // chunk_size,
                    'start_row': i,
                    'end_row': min(i + chunk_size, len(df)),
                    'total_rows': len(chunk_df),
                    'has_labels': 'Label' in chunk_df.columns
                }
            })
        
        return chunks
    
    def ingest_csv_files(self, max_rows_per_file: int = 10000, include_testing: bool = True):
        """
        Ingest all CSV files from data_original/ and Testing_data/ into vector database
        
        Args:
            max_rows_per_file: Maximum rows to process per file (for demo)
            include_testing: Whether to also ingest Testing_data folder
        """
        all_chunks = []
        all_metadatas = []
        all_ids = []
        all_embeddings = []
        
        # Process training data (data_original)
        directories_to_process = [self.data_dir]
        if include_testing and os.path.exists(self.testing_data_dir):
            directories_to_process.append(self.testing_data_dir)
        
        for data_dir in directories_to_process:
            if not os.path.exists(data_dir):
                print(f"Warning: Directory not found: {data_dir}")
                continue
            
            csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            dir_name = os.path.basename(data_dir)
            print(f"\nProcessing {len(csv_files)} CSV files from {dir_name}/...")
            
            if len(csv_files) == 0:
                print(f"No CSV files found in {data_dir}")
                continue
            
            for filename in csv_files:
                print(f"\nProcessing {filename}...")
                filepath = os.path.join(data_dir, filename)
                
                try:
                    # Load CSV (limit rows for demo)
                    df = pd.read_csv(filepath, nrows=max_rows_per_file)
                    print(f"  Loaded {len(df)} rows")
                    
                    # Create chunks
                    chunks = self.chunk_csv_data(df, filename, chunk_size=100)
                    print(f"  Created {len(chunks)} chunks")
                    
                    # Generate embeddings
                    chunk_texts = [chunk['text'] for chunk in chunks]
                    print(f"  Generating embeddings...")
                    embeddings = self.embedder.encode(chunk_texts, show_progress_bar=True)
                    
                    # Prepare for ChromaDB
                    for idx, chunk in enumerate(chunks):
                        # Include directory name in chunk ID to avoid conflicts
                        chunk_id = f"{dir_name}_{filename}_chunk_{chunk['metadata']['chunk_index']}"
                        all_ids.append(chunk_id)
                        chunk['metadata']['source_dir'] = dir_name
                        all_metadatas.append(chunk['metadata'])
                        all_chunks.append(chunk['text'])
                        all_embeddings.append(embeddings[idx].tolist())
                    
                    print(f"  Processed {filename}")
                    
                except Exception as e:
                    print(f"  Error processing {filename}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # Add to ChromaDB collection
        if all_chunks:
            print(f"\nAdding {len(all_chunks)} chunks to vector database...")
            self.collection.add(
                embeddings=all_embeddings,
                documents=all_chunks,
                metadatas=all_metadatas,
                ids=all_ids
            )
            print(f"Successfully ingested {len(all_chunks)} chunks")
        else:
            print("No chunks to ingest")
    
    def get_collection(self):
        """Get the ChromaDB collection"""
        return self.collection

if __name__ == "__main__":
    # Run ingestion
    print("="*60)
    print("RAG Data Ingestion")
    print("="*60)
    
    ingester = RAGDataIngestion()
    ingester.ingest_csv_files(max_rows_per_file=10000, include_testing=True)
    print("\nData ingestion complete!")
    print(f"\nVector database stored in: {os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'chroma_db')}")
