"""
RAG-Powered Q&A Interface for Intrusion Detection Dataset
Self-contained Streamlit application with all RAG functionality
Run with: streamlit run web_demo.py
"""

import streamlit as st
import os
import sys
import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Add parent directory to path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

# ============================================================================
# LLM Wrapper Class
# ============================================================================

class LLMWrapper:
    """Wrapper for LLM using Ollama"""
    
    def __init__(self, model_name: str = "llama3.2:3b", use_ollama: bool = True):
        self.model_name = model_name
        self.use_ollama = use_ollama
        
        if use_ollama:
            try:
                import ollama
                self.ollama_client = ollama
            except ImportError:
                raise ImportError("Ollama not installed. Install with: pip install ollama")
    
    def generate(self, prompt: str, context: Optional[str] = None, 
                max_tokens: int = 500, temperature: float = 0.3) -> str:
        """Generate response from LLM"""
        if context:
            full_prompt = f"""You are an AI assistant helping with network intrusion detection analysis.

CRITICAL INSTRUCTIONS:
- You must ONLY use information from the context provided below
- Do not use any external knowledge or make assumptions beyond what is explicitly stated in the context
- If the context doesn't contain enough information, explicitly state: "The provided context does not contain enough information to answer this question"
- Do not make up or infer information that is not in the context
- Quote specific numbers, statistics, or facts from the context when possible

Context from dataset:
{context}

User Question: {prompt}

Based ONLY on the context above, provide a helpful answer. If the context doesn't contain enough information to answer the question, explicitly state that the context is insufficient."""
        else:
            full_prompt = f"""You are an AI assistant helping with network intrusion detection analysis.

User Question: {prompt}

Please provide a helpful answer."""
        
        if self.use_ollama:
            try:
                response = self.ollama_client.generate(
                    model=self.model_name,
                    prompt=full_prompt,
                    options={
                        'temperature': temperature,
                        'num_predict': max_tokens
                    }
                )
                return response['response']
            except Exception as e:
                return f"Error generating response: {e}\n\nPlease ensure Ollama is running and the model is installed:\n  ollama pull {self.model_name}\n\nRecommended model: llama3.2:3b"

# ============================================================================
# Data Ingestion Functions
# ============================================================================

def chunk_csv_data(df: pd.DataFrame, filename: str, chunk_size: int = 100) -> List[Dict]:
    """Convert DataFrame rows into text chunks for embedding"""
    chunks = []
    
    for i in range(0, len(df), chunk_size):
        chunk_df = df.iloc[i:i+chunk_size]
        
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
        
        # Add sample rows
        chunk_text += "Sample Records:\n"
        sample_rows = chunk_df.head(5).to_string()
        chunk_text += sample_rows + "\n"
        
        # Add key features
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

def ingest_data(data_dir: str, testing_data_dir: str, chroma_dir: str, 
                collection_name: str = 'intrusion_detection_data',
                max_rows_per_file: int = 10000, include_testing: bool = True):
    """Ingest CSV files into vector database"""
    
    print("Loading embedding model...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize ChromaDB
    os.makedirs(chroma_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_dir)
    
    try:
        collection = client.get_collection(name=collection_name)
        print(f"Using existing collection: {collection_name}")
    except:
        collection = client.create_collection(
            name=collection_name,
            metadata={"description": "Intrusion Detection System RAG data"}
        )
        print(f"Created new collection: {collection_name}")
    
    all_chunks = []
    all_metadatas = []
    all_ids = []
    all_embeddings = []
    
    # Process directories
    directories_to_process = [data_dir]
    if include_testing and os.path.exists(testing_data_dir):
        directories_to_process.append(testing_data_dir)
    
    for data_dir_path in directories_to_process:
        if not os.path.exists(data_dir_path):
            print(f"Warning: Directory not found: {data_dir_path}")
            continue
        
        csv_files = [f for f in os.listdir(data_dir_path) if f.endswith('.csv')]
        dir_name = os.path.basename(data_dir_path)
        print(f"\nProcessing {len(csv_files)} CSV files from {dir_name}/...")
        
        for filename in csv_files:
            print(f"Processing {filename}...")
            filepath = os.path.join(data_dir_path, filename)
            
            try:
                df = pd.read_csv(filepath, nrows=max_rows_per_file)
                print(f"  Loaded {len(df)} rows")
                
                chunks = chunk_csv_data(df, filename, chunk_size=100)
                print(f"  Created {len(chunks)} chunks")
                
                chunk_texts = [chunk['text'] for chunk in chunks]
                print(f"  Generating embeddings...")
                embeddings = embedder.encode(chunk_texts, show_progress_bar=True)
                
                for idx, chunk in enumerate(chunks):
                    chunk_id = f"{dir_name}_{filename}_chunk_{chunk['metadata']['chunk_index']}"
                    all_ids.append(chunk_id)
                    chunk['metadata']['source_dir'] = dir_name
                    all_metadatas.append(chunk['metadata'])
                    all_chunks.append(chunk['text'])
                    all_embeddings.append(embeddings[idx].tolist())
                
                print(f"  Processed {filename}")
                
            except Exception as e:
                print(f"  Error processing {filename}: {e}")
                continue
    
    # Add to ChromaDB
    if all_chunks:
        print(f"\nAdding {len(all_chunks)} chunks to vector database...")
        collection.add(
            embeddings=all_embeddings,
            documents=all_chunks,
            metadatas=all_metadatas,
            ids=all_ids
        )
        print(f"Successfully ingested {len(all_chunks)} chunks")
    else:
        print("No chunks to ingest")
    
    return collection

# ============================================================================
# RAG Service Class
# ============================================================================

class RAGService:
    """RAG service for intrusion detection queries"""
    
    def __init__(self, collection_name: str = 'intrusion_detection_data',
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 llm_model: str = "llama3.2:3b",
                 use_ollama: bool = True,
                 top_k: int = 3,
                 chroma_dir: str = None):
        """Initialize RAG service"""
        self.top_k = top_k
        
        # Set ChromaDB directory
        if chroma_dir is None:
            chroma_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chroma_db')
        
        # Initialize embedding model
        self.embedder = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB
        if not os.path.exists(chroma_dir):
            raise ValueError(f"ChromaDB directory not found: {chroma_dir}\nPlease run data ingestion first.")
        
        self.client = chromadb.PersistentClient(path=chroma_dir)
        
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except Exception as e:
            raise ValueError(f"Collection {collection_name} not found. Please run data ingestion first.\nError: {e}")
        
        # Initialize LLM
        self.llm = LLMWrapper(model_name=llm_model, use_ollama=use_ollama)
    
    def retrieve_context(self, query: str) -> List[Dict]:
        """Retrieve relevant context chunks for a query"""
        query_embedding = self.embedder.encode(query).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k
        )
        
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
        """Process a user query with RAG"""
        contexts = self.retrieve_context(user_query)
        
        context_text = "\n\n---\n\n".join([
            f"Context {i+1} (from {ctx['metadata'].get('filename', 'unknown')}):\n{ctx['text']}"
            for i, ctx in enumerate(contexts)
        ])
        
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

# ============================================================================
# Streamlit Application
# ============================================================================

st.set_page_config(
    page_title="Intrusion Detection System - RAG Demo",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=None
)

st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    /* Fix chat input box color - prevent red border */
    .stChatInput input,
    .stChatInput input:focus,
    .stChatInput input:hover,
    .stChatInput input:active,
    .stChatInput > div > div > input,
    .stChatInput > div > div > input:focus,
    .stChatInput > div > div > input:hover,
    .stChatInput > div > div > input:active {
        border-color: #cccccc !important;
        border: 1px solid #cccccc !important;
        outline: none !important;
        box-shadow: none !important;
    }
    .stChatInput input:focus,
    .stChatInput > div > div > input:focus {
        border-color: #1f77b4 !important;
        border: 1px solid #1f77b4 !important;
        box-shadow: 0 0 0 0.2rem rgba(31, 119, 180, 0.25) !important;
    }
    /* Override any error/red states and container borders */
    .stChatInput > div,
    .stChatInput > div > div {
        border-color: transparent !important;
    }
    .stChatInput input[aria-invalid="true"],
    .stChatInput input.error,
    .stChatInput > div > div > input[aria-invalid="true"] {
        border-color: #cccccc !important;
        border: 1px solid #cccccc !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_rag_service():
    """Load RAG service (cached)"""
    try:
        rag = RAGService(use_ollama=True)
        return rag
    except Exception as e:
        return None, str(e)

def main():
    st.markdown('<div class="main-header">Network Intrusion Detection System</div>', unsafe_allow_html=True)
    
    # Load RAG service
    rag_result = load_rag_service()
    
    if isinstance(rag_result, tuple):
        rag_service = None
        error_msg = rag_result[1]
    else:
        rag_service = rag_result
        error_msg = None
    
    if rag_service is None:
        st.error("RAG service is not available. Please set it up first.")
        st.info("""
        **Quick Setup:**
        1. Install dependencies: `pip install -r requirements.txt`
        2. Install Ollama: https://ollama.ai
        3. Pull model: `ollama pull llama3.2:3b`
        4. Start Ollama: `ollama serve`
        5. Run data ingestion: `python web_demo.py --ingest`
        6. Refresh this page
        """)
        st.stop()
    
    # Small clear button in top right corner
    col1, col2 = st.columns([10, 1])
    with col2:
        if st.button("🗑️", help="Clear chat history", key="clear_chat_btn"):
            st.session_state.rag_messages = []
            st.rerun()
    
    # Initialize chat history
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []
    
    # Display chat history with unique keys
    for msg_idx, message in enumerate(st.session_state.rag_messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "contexts" in message and len(message.get("contexts", [])) > 0:
                with st.expander(f"Sources ({len(message['contexts'])} chunks)"):
                    for i, ctx in enumerate(message['contexts']):
                        filename = ctx['metadata'].get('filename', 'unknown')
                        start_row = ctx['metadata'].get('start_row', '?')
                        end_row = ctx['metadata'].get('end_row', '?')
                        source_dir = ctx['metadata'].get('source_dir', '')
                        
                        if source_dir:
                            st.markdown(f"**Source {i+1}:** `{source_dir}/{filename}`")
                        else:
                            st.markdown(f"**Source {i+1}:** `{filename}`")
                        st.caption(f"Rows {start_row}-{end_row}")
                        
                        # Use unique key combining message index and context index
                        checkbox_key = f"ctx_hist_{msg_idx}_{i}"
                        text_key = f"text_hist_{msg_idx}_{i}"
                        if st.checkbox(f"Show context {i+1}", key=checkbox_key):
                            st.text_area("", ctx['text'], height=200, key=text_key, disabled=True)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the dataset..."):
        st.session_state.rag_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                try:
                    result = rag_service.query(prompt)
                    
                    st.markdown(result['answer'])
                    
                    if result['contexts']:
                        with st.expander(f"Sources ({len(result['contexts'])} chunks)"):
                            for i, ctx in enumerate(result['contexts']):
                                filename = ctx['metadata'].get('filename', 'unknown')
                                start_row = ctx['metadata'].get('start_row', '?')
                                end_row = ctx['metadata'].get('end_row', '?')
                                source_dir = ctx['metadata'].get('source_dir', '')
                                
                                if source_dir:
                                    st.markdown(f"**Source {i+1}:** `{source_dir}/{filename}`")
                                else:
                                    st.markdown(f"**Source {i+1}:** `{filename}`")
                                st.caption(f"Rows {start_row}-{end_row}")
                                
                                # Use unique key for new results
                                checkbox_key = f"ctx_result_{len(st.session_state.rag_messages)}_{i}"
                                text_key = f"text_result_{len(st.session_state.rag_messages)}_{i}"
                                if st.checkbox(f"Show context {i+1}", key=checkbox_key):
                                    st.text_area("", ctx['text'], height=200, key=text_key, disabled=True)
                    
                    st.session_state.rag_messages.append({
                        "role": "assistant",
                        "content": result['answer'],
                        "contexts": result.get('contexts', [])
                    })
                except Exception as e:
                    st.error(f"Error generating response: {e}")
                    import traceback
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())
    

if __name__ == "__main__":
    # Check if running data ingestion from command line
    if len(sys.argv) > 1 and sys.argv[1] == '--ingest':
        print("="*60)
        print("RAG Data Ingestion")
        print("="*60)
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, 'CSE543_Group1', 'data_original')
        testing_data_dir = os.path.join(base_dir, 'CSE543_Group1', 'Testing_data')
        chroma_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chroma_db')
        
        collection = ingest_data(data_dir, testing_data_dir, chroma_dir)
        print(f"\nData ingestion complete! {collection.count()} chunks ingested.")
    else:
        # Run Streamlit app
        main()

