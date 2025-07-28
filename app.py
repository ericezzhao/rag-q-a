"""
RAG Q&A System - Streamlit Application
Main entry point for the RAG Q&A system using ChromaDB, LlamaIndex, and Streamlit
"""

import streamlit as st
import tempfile
from pathlib import Path
import time

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent / "src"))

from rag_pipeline import RAGPipeline
from config import Config


@st.cache_resource
def initialize_rag_pipeline():
    """Initialize RAG pipeline with caching"""
    try:
        # Use data directory for persistence
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        pipeline = RAGPipeline(
            vector_store_path=str(data_dir / "vector_store"),
            collection_name="streamlit_rag_collection"
        )
        return pipeline, None
    except Exception as e:
        return None, str(e)


def display_system_status(pipeline):
    """Display system status in sidebar"""
    st.sidebar.markdown("### 🔍 System Status")
    
    try:
        status = pipeline.get_system_status()
        
        # Vector store status
        vector_status = status['vector_store']['status']
        if vector_status == 'healthy':
            st.sidebar.success(f"🗄️ Vector Store: {vector_status}")
            
            # Get actual file count instead of chunk count
            try:
                doc_list = pipeline.list_documents()
                if doc_list['success']:
                    file_count = doc_list['total_sources']
                    chunk_count = doc_list['total_documents']
                    st.sidebar.write(f"📄 Files: {file_count}")
                    st.sidebar.write(f"🧩 Chunks: {chunk_count}")
                else:
                    st.sidebar.write(f"📄 Files: 0")
            except:
                # Fallback to old method if there's an error
                doc_count = status['vector_store'].get('document_count', 0)
                st.sidebar.write(f"🧩 Chunks: {doc_count}")
        else:
            st.sidebar.error(f"🗄️ Vector Store: {vector_status}")
        
        # LLM status
        llm_status = status['llm_service']['status']
        if llm_status == 'healthy':
            st.sidebar.success(f"🤖 LLM: {llm_status}")
            st.sidebar.write(f"🧠 Model: {status['llm_service']['model']}")
        else:
            st.sidebar.error(f"🤖 LLM: {llm_status}")
        
        # Pipeline status
        pipeline_status = status['pipeline']['status']
        if pipeline_status == 'operational':
            st.sidebar.success(f"🔄 Pipeline: {pipeline_status}")
            ready_for_queries = status['pipeline']['ready_for_queries']
            st.sidebar.write(f"❓ Ready for queries: {'✅' if ready_for_queries else '❌'}")
        else:
            st.sidebar.error(f"🔄 Pipeline: {pipeline_status}")
            
    except Exception as e:
        st.sidebar.error(f"❌ Status check failed: {str(e)}")


def document_upload_interface(pipeline):
    """Document upload and processing interface"""
    st.header("📄 Document Management")
    
    # Create tabs for upload and management
    upload_tab, manage_tab = st.tabs(["📤 Upload Documents", "📋 Manage Documents"])
    
    with upload_tab:
        # File upload
        uploaded_files = st.file_uploader(
            "Upload documents for Q&A",
            type=['pdf', 'txt', 'docx'],
            accept_multiple_files=True,
            help="Upload PDF, TXT, or DOCX files to add to your knowledge base"
        )
        
        if uploaded_files:
            # Queue header with clear button
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader("📋 Processing Queue")
            with col2:
                if st.button("🗑️ Clear Queue", help="Clear the upload queue"):
                    st.session_state.processed_files = set()
                    st.rerun()
            
            # Initialize session state for processed files
            if 'processed_files' not in st.session_state:
                st.session_state.processed_files = set()
            
            # Filter out already processed files
            pending_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
            
            if not pending_files:
                st.success("✅ All uploaded files have been processed!")
                st.info("💡 Upload more files or switch to the 'Manage Documents' tab")
            else:
                # Process each pending file
                for uploaded_file in pending_files:
                    with st.expander(f"📄 {uploaded_file.name}", expanded=True):
                        # File info
                        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
                        st.write(f"📊 **File size:** {file_size_mb:.2f} MB")
                        st.write(f"📝 **File type:** {uploaded_file.type}")
                        
                        # Process button
                        if st.button(f"🔄 Process {uploaded_file.name}", key=f"process_{uploaded_file.name}"):
                            process_file(pipeline, uploaded_file)
    
    with manage_tab:
        display_document_manager(pipeline)


def process_file(pipeline, uploaded_file):
    """Process a single uploaded file"""
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        # Processing indicator
        with st.spinner(f"Processing {uploaded_file.name}..."):
            start_time = time.time()
            
            # Process with RAG pipeline
            result = pipeline.ingest_document(tmp_file_path)
            
            processing_time = time.time() - start_time
        
        if result['success']:
            st.success(f"✅ Successfully processed {uploaded_file.name}")
            
            # Display processing statistics
            stats = result['processing_stats']
            vector_stats = result['vector_store_stats']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📄 Documents", stats['documents_loaded'])
                st.metric("🧩 Chunks", stats['chunks_created'])
            
            with col2:
                st.metric("📝 Characters", f"{stats['total_characters']:,}")
                st.metric("📏 Avg Chunk Size", f"{stats['average_chunk_size']:.0f}")
            
            with col3:
                st.metric("🗄️ Collection Size", vector_stats['total_collection_size'])
                st.metric("⏱️ Processing Time", f"{processing_time:.2f}s")
            
            # Success message
            st.success(f"🎉 Document added to knowledge base! You can now ask questions about {uploaded_file.name}")
            
            # Mark file as processed
            if 'processed_files' not in st.session_state:
                st.session_state.processed_files = set()
            st.session_state.processed_files.add(uploaded_file.name)
            
            # Auto-refresh to remove from processing queue
            time.sleep(1)  # Brief pause so user can see the success message
            st.rerun()
            
        else:
            st.error(f"❌ Failed to process {uploaded_file.name}")
            st.error(f"**Error:** {result['error']}")
            
    except Exception as e:
        st.error(f"❌ Processing failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_file_path)
        except:
            pass


def qa_interface(pipeline):
    """Enhanced Q&A interface for querying documents"""
    st.header("💬 Ask Questions")
    
    # Initialize session state for conversation history
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'show_advanced_options' not in st.session_state:
        st.session_state.show_advanced_options = False
    
    # Check if documents are available
    try:
        doc_list = pipeline.list_documents()
        
        if not doc_list['success'] or doc_list['total_sources'] == 0:
            st.warning("📄 No documents in knowledge base. Upload documents first!")
            return
        
        file_count = doc_list['total_sources']
        chunk_count = doc_list['total_documents']
        st.info(f"📚 Knowledge base contains {file_count} files ({chunk_count} chunks). Ask me anything!")
        
    except Exception as e:
        st.error(f"❌ Unable to check document status: {str(e)}")
        return
    

    
    # Main query input
    st.subheader("❓ Your Question")
    
    # Initialize question value in session state
    if 'question_text' not in st.session_state:
        st.session_state.question_text = ''
    
    # Use suggested question if available
    if st.session_state.get('suggested_question', ''):
        st.session_state.question_text = st.session_state.suggested_question
        del st.session_state.suggested_question
    
    question = st.text_area(
        "Enter your question:",
        value=st.session_state.question_text,
        placeholder="e.g., What is this document about?",
        help="Ask any question about the uploaded documents. You can ask multiple questions at once.",
        height=100,
        key="question_input"
    )
    
    # Advanced query settings - always visible
    with st.expander("⚙️ Advanced Query Settings", expanded=True):
        adv_col1, adv_col2, adv_col3 = st.columns(3)
        
        with adv_col1:
            top_k = st.slider("📊 Chunks to retrieve", 1, 10, 5, help="More chunks = more context but slower")
            
        with adv_col2:
            temperature = st.slider("🌡️ Response creativity", 0.0, 1.0, 0.1, 0.1, 
                                  help="Higher = more creative, Lower = more focused")
            
        with adv_col3:
            max_tokens = st.slider("📝 Max response length", 100, 2000, 1000, 50,
                                 help="Maximum tokens in the response")
    
    # Query action buttons
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        ask_button = st.button("🔍 Ask Question", type="primary", use_container_width=True)
    with col2:
        if st.button("🗑️ Clear History", help="Clear conversation history"):
            st.session_state.conversation_history = []
            st.rerun()
    with col3:
        if st.session_state.conversation_history:
            if st.button("📥 Export Chat", help="Export conversation history"):
                export_conversation()
    
    # Process query
    if ask_button and question.strip():
        st.session_state.question_text = ''
        
        with st.spinner("🔍 Searching knowledge base and generating response..."):
            # Store query parameters for potential pipeline enhancement
            query_params = {
                'top_k': top_k,
                'temperature': temperature,
                'max_tokens': max_tokens
            }
            
            result = pipeline.query(question, top_k=top_k)
        
        if result['success']:
            # Add to conversation history
            conversation_entry = {
                'timestamp': time.time(),
                'question': question,
                'response': result['response'],
                'sources': result['retrieved_chunks'],
                'stats': result['pipeline_stats'],
                'query_params': query_params
            }
            st.session_state.conversation_history.append(conversation_entry)
            
            # Display the latest response with enhanced formatting
            display_enhanced_response(conversation_entry, len(st.session_state.conversation_history))
            
            # Rerun to update the text area
            st.rerun()
                
        else:
            st.error(f"❌ Query failed: {result['error']}")
            # Restore the question text if there was an error
            st.session_state.question_text = question
    
    elif ask_button and not question.strip():
        st.warning("⚠️ Please enter a question!")
    
    # Display conversation history
    if st.session_state.conversation_history:
        st.markdown("---")
        st.subheader("💬 Conversation History")
        
        # Reverse chronological order (newest first)
        for i, entry in enumerate(reversed(st.session_state.conversation_history)):
            display_enhanced_response(entry, len(st.session_state.conversation_history) - i, is_history=True)


def display_enhanced_response(entry, entry_number, is_history=False):
    """Display an enhanced, well-formatted response"""
    import datetime
    
    timestamp = datetime.datetime.fromtimestamp(entry['timestamp']).strftime("%H:%M:%S")
    
    # Response container with better styling
    with st.container():
        # Header with question and metadata
        if is_history:
            header_text = f"**Q{entry_number}** ({timestamp})"
        else:
            header_text = f"**Latest Response** ({timestamp})"
        
        st.markdown(f"### {header_text}")
        
        # Question in a nice box
        st.markdown("**❓ Question:**")
        st.info(entry['question'])
        
        # Answer with enhanced formatting
        st.markdown("**🤖 Answer:**")
        
        # Create a bordered container for the answer
        answer_container = st.container()
        with answer_container:
            # Custom CSS for better answer display
            st.markdown(f"""
            <div style="
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                border-left: 4px solid #007bff;
                margin: 10px 0;
            ">
                {entry['response']}
            </div>
            """, unsafe_allow_html=True)
        
        # Action buttons for this response
        action_col1, action_col2, action_col3, action_col4 = st.columns(4)
        
        # Create unique keys using entry number, timestamp, and history flag
        unique_id = f"{entry_number}_{int(entry['timestamp'] * 1000000)}_{is_history}"
        
        with action_col1:
            if st.button(f"📋 Copy Response", key=f"copy_{unique_id}", help="Copy response to clipboard"):
                copy_to_clipboard_js(entry['response'])
        
        with action_col2:
            show_sources_key = f"show_sources_{unique_id}"
            if st.button(f"📚 Sources ({len(entry['sources'])})", key=show_sources_key, help="Show/hide source documents"):
                # Toggle sources display
                toggle_key = f"sources_visible_{unique_id}"
                st.session_state[toggle_key] = not st.session_state.get(toggle_key, False)
        
        with action_col3:
            stats_key = f"show_stats_{unique_id}"
            if st.button(f"📊 Stats", key=stats_key, help="Show/hide query statistics"):
                # Toggle stats display
                toggle_key = f"stats_visible_{unique_id}"
                st.session_state[toggle_key] = not st.session_state.get(toggle_key, False)
        
        with action_col4:
            if st.button(f"🔄 Retry Query", key=f"retry_{unique_id}", help="Ask this question again"):
                st.session_state.question_text = entry['question']
                st.rerun()
        
        # Show sources if toggled
        sources_toggle_key = f"sources_visible_{unique_id}"
        if st.session_state.get(sources_toggle_key, False):
            display_sources_enhanced(entry['sources'])
        
        # Show stats if toggled
        stats_toggle_key = f"stats_visible_{unique_id}"
        if st.session_state.get(stats_toggle_key, False):
            display_stats_enhanced(entry['stats'], entry['query_params'])
        
        if not is_history:
            st.markdown("---")
        else:
            st.markdown("<hr style='margin: 15px 0; border: 1px solid #e0e0e0;'>", unsafe_allow_html=True)


def display_sources_enhanced(sources):
    """Display sources with enhanced formatting"""
    st.markdown("**📚 Source Documents:**")
    
    for i, chunk in enumerate(sources, 1):
        metadata = chunk['metadata']
        file_name = metadata.get('file_name', f'Source {i}')
        section = metadata.get('section', 'unknown section')
        similarity = chunk['similarity_score']
        
        with st.expander(f"📄 {file_name} ({section}) - Relevance: {similarity:.1%}", expanded=False):
            # Content with better formatting
            st.markdown("**📝 Content:**")
            st.markdown(f"```\n{chunk['text']}\n```")
            
            # Metadata in a clean format
            st.markdown("**📋 Document Details:**")
            metadata_cols = st.columns(2)
            
            for idx, (key, value) in enumerate(metadata.items()):
                col_idx = idx % 2
                with metadata_cols[col_idx]:
                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")


def display_stats_enhanced(stats, query_params):
    """Display query statistics with enhanced formatting"""
    st.markdown("**📊 Query Performance:**")
    
    # Performance metrics
    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    
    with perf_col1:
        st.metric("🧩 Chunks Retrieved", stats['chunks_retrieved'])
    with perf_col2:
        st.metric("📝 Context Length", f"{stats['total_context_chars']:,} chars")
    with perf_col3:
        st.metric("⏱️ Query Time", f"{stats['query_time_seconds']:.2f}s")
    with perf_col4:
        avg_similarity = sum(chunk['similarity_score'] for chunk in stats.get('retrieved_chunks', [])) / max(len(stats.get('retrieved_chunks', [])), 1)
        st.metric("🎯 Avg Relevance", f"{avg_similarity:.1%}")
    
    # Query parameters used
    st.markdown("**⚙️ Query Parameters:**")
    param_cols = st.columns(3)
    
    with param_cols[0]:
        st.write(f"**Chunks Retrieved:** {query_params['top_k']}")
    with param_cols[1]:
        st.write(f"**Temperature:** {query_params['temperature']}")
    with param_cols[2]:
        st.write(f"**Max Tokens:** {query_params['max_tokens']}")


def copy_to_clipboard_js(text):
    """Generate JavaScript to copy text to clipboard"""
    st.markdown(f"""
    <script>
    navigator.clipboard.writeText(`{text}`).then(function() {{
        console.log('Text copied to clipboard');
    }});
    </script>
    """, unsafe_allow_html=True)
    st.success("📋 Response copied to clipboard!")


def export_conversation():
    """Export conversation history"""
    import datetime
    
    if not st.session_state.conversation_history:
        st.warning("No conversation history to export")
        return
    
    # Generate export content
    export_content = "# RAG Q&A Conversation Export\n\n"
    export_content += f"**Export Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    export_content += f"**Total Questions:** {len(st.session_state.conversation_history)}\n\n"
    export_content += "---\n\n"
    
    for i, entry in enumerate(st.session_state.conversation_history, 1):
        timestamp = datetime.datetime.fromtimestamp(entry['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
        export_content += f"## Question {i} ({timestamp})\n\n"
        export_content += f"**Q:** {entry['question']}\n\n"
        export_content += f"**A:** {entry['response']}\n\n"
        export_content += f"**Sources:** {len(entry['sources'])} documents\n\n"
        export_content += "---\n\n"
    
    # Offer download
    st.download_button(
        label="📥 Download Conversation",
        data=export_content,
        file_name=f"rag_conversation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        help="Download the conversation history as a Markdown file"
    )


def display_document_manager(pipeline):
    """Display document management interface"""
    st.subheader("📋 Knowledge Base Documents")
    
    # Get list of documents
    try:
        doc_list = pipeline.list_documents()
        
        if not doc_list['success']:
            st.error(f"❌ Failed to load documents: {doc_list['error']}")
            return
        
        if doc_list['total_sources'] == 0:
            st.info("📄 No documents in knowledge base. Upload some documents first!")
            return
        
        # Display document statistics
        st.info(f"📊 Knowledge base contains {doc_list['total_sources']} files with {doc_list['total_documents']} total chunks")
        
        # Create columns for layout
        col1, col2 = st.columns([3, 1])
        
        
        with col2:
            # Clear all button
            if st.button("🗑️ Clear All", type="secondary", help="Remove all documents from knowledge base"):
                if st.session_state.get('confirm_clear_all', False):
                    with st.spinner("Clearing knowledge base..."):
                        success = pipeline.clear_vector_store()
                    if success:
                        st.success("✅ Knowledge base cleared successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to clear knowledge base")
                    st.session_state['confirm_clear_all'] = False
                else:
                    st.session_state['confirm_clear_all'] = True
                    st.warning("⚠️ Click again to confirm clearing ALL documents")
        
        # Display each document with remove option
        for file_name, chunk_count in doc_list['source_counts'].items():
            with st.container():
                doc_col1, doc_col2, doc_col3 = st.columns([2, 1, 1])
                
                with doc_col1:
                    st.write(f"📄 **{file_name}**")
                
                with doc_col2:
                    st.write(f"🧩 {chunk_count} chunks")
                
                with doc_col3:
                    # Remove button for individual document
                    if st.button(f"🗑️ Remove", key=f"remove_{file_name}", help=f"Remove {file_name} from knowledge base"):
                        with st.spinner(f"Removing {file_name}..."):
                            result = pipeline.remove_document(file_name)
                        
                        if result['success']:
                            st.success(f"✅ {result['message']}")
                            st.rerun()
                        else:
                            st.error(f"❌ Failed to remove {file_name}: {result['error']}")
                
                st.divider()
    
    except Exception as e:
        st.error(f"❌ Error in document manager: {str(e)}")


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="RAG Q&A System",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 RAG Q&A System")
    st.markdown("Built with ChromaDB, LlamaIndex, and OpenAI GPT-4o")
    st.markdown("---")
    
    # Initialize RAG pipeline
    pipeline, error = initialize_rag_pipeline()
    
    if error:
        st.error(f"❌ Failed to initialize RAG pipeline: {error}")
        st.info("💡 Make sure your OpenAI API key is configured in the environment")
        return
    
    # Display system status in sidebar
    display_system_status(pipeline)
    
    # Main interface tabs
    tab1, tab2 = st.tabs(["📄 Document Management", "💬 Q&A Interface"])
    
    with tab1:
        document_upload_interface(pipeline)
    
    with tab2:
        qa_interface(pipeline)


if __name__ == "__main__":
    main() 