"""
RAG-Powered Q&A Interface for Intrusion Detection Dataset
Interactive Streamlit application for querying network intrusion detection data
"""

import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import RAG components
from rag_service import RAGService

# Page config
st.set_page_config(
    page_title="Intrusion Detection System - RAG Demo",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=None
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
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
    # Header
    st.markdown('<div class="main-header">Network Intrusion Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; color: #666; margin-bottom: 2rem;">RAG-Powered Q&A Interface</div>', unsafe_allow_html=True)
    
    # Load RAG service
    rag_result = load_rag_service()
    
    if isinstance(rag_result, tuple):
        rag_service = None
        error_msg = rag_result[1]
    else:
        rag_service = rag_result
        error_msg = None
    
    # Sidebar
    with st.sidebar:
        st.header("System Status")
        if rag_service:
            st.success("RAG Service: Active")
            try:
                collection = rag_service.collection
                st.info(f"Data Chunks: {collection.count()}")
            except:
                pass
        else:
            st.error("RAG Service: Not Available")
        
        st.markdown("---")
        st.header("Setup Instructions")
        with st.expander("How to Enable RAG"):
            st.markdown("""
            ### Setup RAG System:
            
            1. **Install dependencies:**
               ```bash
               pip install -r requirements.txt
               ```
            
            2. **Install Ollama and pull Phi-3:**
               ```bash
               # Install Ollama from https://ollama.ai
               ollama pull phi3:mini
               ```
            
            3. **Run data ingestion:**
               ```bash
               python src/rag_data_ingestion.py
               ```
            
            4. **Restart this demo**
            """)
        
        if error_msg:
            st.error(f"Error: {error_msg}")
    
    # Main content
    st.header("Query Dataset")
    st.markdown("Ask questions about the network intrusion detection data. The system will retrieve relevant information from CSV files and provide answers.")
    
    if rag_service is None:
        st.error("RAG service is not available. Please set it up first.")
        st.info("""
        **Quick Setup:**
        1. Install dependencies: `pip install -r requirements.txt`
        2. Install Ollama: https://ollama.ai
        3. Pull model: `ollama pull phi3:mini`
        4. Run ingestion: `python src/rag_data_ingestion.py`
        5. Restart this app
        """)
        st.stop()
    
    # Initialize chat history
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []
    
    # Display chat history
    for message in st.session_state.rag_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show context sources if available
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
                        
                        if st.checkbox(f"Show context {i+1}", key=f"ctx_{len(st.session_state.rag_messages)}_{i}"):
                            st.text_area("", ctx['text'], height=200, key=f"text_{len(st.session_state.rag_messages)}_{i}", disabled=True)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the dataset..."):
        # Add user message
        st.session_state.rag_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                try:
                    result = rag_service.query(prompt)
                    
                    st.markdown(result['answer'])
                    
                    # Show context sources
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
                                
                                if st.checkbox(f"Show context {i+1}", key=f"ctx_result_{i}"):
                                    st.text_area("", ctx['text'], height=200, key=f"text_result_{i}", disabled=True)
                    
                    # Add assistant message to history
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
    
    # Clear chat button
    st.markdown("---")
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.rag_messages = []
        st.rerun()

if __name__ == "__main__":
    main()
