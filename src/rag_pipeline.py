"""
RAG Pipeline Module for Q&A System
Integrates document processing, vector storage, and LLM services into a complete RAG pipeline
Built using LlamaIndex lower-level abstractions
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import time

from llama_index.core.schema import TextNode

# fix import issues
try:
    from .document_processor import DocumentProcessor
    from .vector_store import ChromaVectorStore
    from .llm_service import LLMService, RAGContext
    from .config import Config
except ImportError:
    from document_processor import DocumentProcessor
    from vector_store import ChromaVectorStore
    from llm_service import LLMService, RAGContext
    from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Complete RAG pipeline integrating all components
    Handles end-to-end document ingestion, retrieval, and response generation
    """
    
    def __init__(self, vector_store_path: Optional[str] = None, collection_name: Optional[str] = None):
        """
        Initialize RAG pipeline with all components
        
        Args:
            vector_store_path: Path for ChromaDB persistence
            collection_name: Name of ChromaDB collection
        """
        self.vector_store_path = vector_store_path or Config.CHROMA_DB_PATH
        self.collection_name = collection_name or Config.CHROMA_COLLECTION_NAME
        
        logger.info("Initializing RAG pipeline components...")
        
        # processor, vector store, llm service
        self.document_processor = DocumentProcessor()
        self.vector_store = ChromaVectorStore(
            collection_name=self.collection_name,
            persist_path=self.vector_store_path
        )
        self.llm_service = LLMService()
        
        logger.info("RAG pipeline initialized successfully")
    
    def ingest_document(self, file_path: str) -> Dict[str, Any]:
        """
        Ingest a document into the RAG system
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with ingestion results
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting document ingestion: {file_path}")
            
            # Step 1: Validate file
            validation = self.document_processor.validate_file(file_path)
            if not validation['valid']:
                return {
                    'success': False,
                    'error': f"File validation failed: {validation['errors']}",
                    'file_path': file_path
                }
            
            # Step 2: Process document (load and chunk)
            logger.info("Processing document...")
            processing_result = self.document_processor.process_uploaded_file(file_path)
            
            if not processing_result['success']:
                return {
                    'success': False,
                    'error': f"Document processing failed: {processing_result['error']}",
                    'file_path': file_path
                }
            
            chunks = processing_result['chunks']
            
            # Step 3: Add chunks to vector store
            logger.info("Adding chunks to vector store...")
            vector_result = self.vector_store.add_nodes(chunks)
            
            if not vector_result['success']:
                return {
                    'success': False,
                    'error': f"Vector store ingestion failed: {vector_result['error']}",
                    'file_path': file_path
                }
            
            # Step 4: Compile results
            ingestion_time = time.time() - start_time
            
            result = {
                'success': True,
                'file_path': file_path,
                'processing_stats': {
                    'documents_loaded': processing_result['documents_loaded'],
                    'chunks_created': processing_result['chunks_created'],
                    'total_characters': processing_result['total_characters'],
                    'average_chunk_size': processing_result['average_chunk_size']
                },
                'vector_store_stats': {
                    'nodes_added': vector_result['nodes_added'],
                    'total_collection_size': vector_result['collection_count']
                },
                'metadata': processing_result['metadata'],
                'ingestion_time_seconds': round(ingestion_time, 2)
            }
            
            logger.info(f"Document ingestion completed successfully in {ingestion_time:.2f}s")
            logger.info(f"Added {vector_result['nodes_added']} chunks. Collection now has {vector_result['collection_count']} documents")
            
            return result
            
        except Exception as e:
            error_msg = f"Error during document ingestion: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'file_path': file_path,
                'ingestion_time_seconds': time.time() - start_time
            }
    
    def query(self, question: str, top_k: Optional[int] = None, filters: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute end-to-end RAG query
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            filters: Optional metadata filters for retrieval
            
        Returns:
            Dictionary with complete RAG response
        """
        start_time = time.time()
        top_k = top_k or Config.MAX_RETRIEVED_CHUNKS
        
        try:
            logger.info(f"Processing RAG query: '{question[:50]}...'")
            
            # Step 1: Retrieve relevant chunks
            logger.info("Retrieving relevant chunks...")
            retrieval_result = self.vector_store.query(
                query_text=question,
                top_k=top_k,
                filters=filters
            )
            
            if not retrieval_result['success']:
                return {
                    'success': False,
                    'error': f"Retrieval failed: {retrieval_result['error']}",
                    'query': question
                }
            
            retrieved_chunks = retrieval_result['nodes']
            
            if not retrieved_chunks:
                return {
                    'success': True,
                    'query': question,
                    'response': "I don't have any relevant information in my knowledge base to answer your question.",
                    'retrieved_chunks': [],
                    'sources': [],
                    'query_time_seconds': time.time() - start_time
                }
            
            # Step 2: Prepare RAG context
            context_text = "\n\n".join([chunk['text'] for chunk in retrieved_chunks])
            
            rag_context = RAGContext(
                query=question,
                retrieved_chunks=retrieved_chunks,
                context_text=context_text,
                metadata={
                    'retrieval_method': 'vector_similarity',
                    'collection_size': retrieval_result['collection_size'],
                    'filters_applied': filters or {}
                }
            )
            
            # Step 3: Generate response
            logger.info("Generating LLM response...")
            llm_result = self.llm_service.generate_response(rag_context)
            
            if not llm_result['success']:
                return {
                    'success': False,
                    'error': f"LLM generation failed: {llm_result['error']}",
                    'query': question,
                    'retrieved_chunks': retrieved_chunks
                }
            
            # Step 4: Compile complete response
            query_time = time.time() - start_time
            
            result = {
                'success': True,
                'query': question,
                'response': llm_result['response'],
                'retrieved_chunks': [{
                    'text': chunk['text'],
                    'metadata': chunk['metadata'],
                    'similarity_score': chunk['score']
                } for chunk in retrieved_chunks],
                'sources': self._extract_sources(retrieved_chunks),
                'pipeline_stats': {
                    'chunks_retrieved': len(retrieved_chunks),
                    'total_context_chars': len(context_text),
                    'collection_size': retrieval_result['collection_size'],
                    'query_time_seconds': round(query_time, 2),
                    'model_used': llm_result['model_info']['model']
                },
                'metadata': {
                    'filters_used': filters,
                    'top_k_requested': top_k,
                    'retrieval_successful': True,
                    'generation_successful': True
                }
            }
            
            logger.info(f"RAG query completed successfully in {query_time:.2f}s")
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks, generated {len(llm_result['response'])} char response")
            
            return result
            
        except Exception as e:
            error_msg = f"Error during RAG query: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'query': question,
                'query_time_seconds': time.time() - start_time
            }
    
    def chat_query(self, messages: List[Dict[str, str]], use_context: bool = True, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute chat-style RAG query with conversation history
        
        Args:
            messages: List of chat messages [{'role': 'user/assistant', 'content': '...'}]
            use_context: Whether to retrieve and use context
            top_k: Number of chunks to retrieve
            
        Returns:
            Dictionary with chat response
        """
        try:
            # Get the latest user message for context retrieval
            latest_message = None
            for msg in reversed(messages):
                if msg['role'] == 'user':
                    latest_message = msg['content']
                    break
            
            if not latest_message:
                return {
                    'success': False,
                    'error': 'No user message found in conversation'
                }
            
            context_chunks = None
            
            # Retrieve context if requested
            if use_context:
                retrieval_result = self.vector_store.query(
                    query_text=latest_message,
                    top_k=top_k or Config.MAX_RETRIEVED_CHUNKS
                )
                
                if retrieval_result['success'] and retrieval_result['nodes']:
                    context_chunks = retrieval_result['nodes']
            
            # Generate chat response
            chat_result = self.llm_service.generate_chat_response(messages, context_chunks)
            
            if chat_result['success']:
                # Add context information to result
                chat_result['context_info'] = {
                    'context_used': bool(context_chunks),
                    'chunks_retrieved': len(context_chunks) if context_chunks else 0,
                    'sources': self._extract_sources(context_chunks) if context_chunks else []
                }
            
            return chat_result
            
        except Exception as e:
            error_msg = f"Error during chat query: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            # Vector store info
            vector_info = self.vector_store.get_collection_info()
            
            # Test LLM connection
            llm_test = self.llm_service.test_connection()
            
            status = {
                'vector_store': {
                    'status': 'healthy' if 'error' not in vector_info else 'error',
                    'collection_name': vector_info.get('collection_name'),
                    'document_count': vector_info.get('document_count', 0),
                    'persist_path': vector_info.get('persist_path'),
                    'embedding_model': vector_info.get('embedding_model')
                },
                'llm_service': {
                    'status': 'healthy' if llm_test['success'] else 'error',
                    'model': Config.DEFAULT_LLM_MODEL,
                    'temperature': Config.LLM_TEMPERATURE,
                    'test_result': llm_test.get('message', llm_test.get('error'))
                },
                'document_processor': {
                    'status': 'healthy',
                    'supported_formats': self.document_processor.get_supported_extensions(),
                    'chunk_size': Config.CHUNK_SIZE,
                    'chunk_overlap': Config.CHUNK_OVERLAP
                },
                'pipeline': {
                    'status': 'operational',
                    'ready_for_ingestion': True,
                    'ready_for_queries': vector_info.get('document_count', 0) > 0
                }
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {
                'error': str(e),
                'status': 'error'
            }
    
    def search_by_source(self, source_filter: str, query: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
        """
        Search documents by source file with optional text query
        
        Args:
            source_filter: Source file name to filter by
            query: Optional text query for semantic search within source
            limit: Maximum results to return
            
        Returns:
            Search results
        """
        try:
            if query:
                # Semantic search with source filter
                result = self.vector_store.query(
                    query_text=query,
                    top_k=limit,
                    filters={'file_name': source_filter}
                )
                
                return {
                    'success': True,
                    'search_type': 'semantic_with_source_filter',
                    'source_filter': source_filter,
                    'query': query,
                    'results': result['nodes'] if result['success'] else [],
                    'num_results': result['num_results'] if result['success'] else 0
                }
            else:
                # Metadata-only search
                result = self.vector_store.search_by_metadata(
                    metadata_filters={'file_name': source_filter},
                    limit=limit
                )
                
                return {
                    'success': True,
                    'search_type': 'metadata_only',
                    'source_filter': source_filter,
                    'results': result['documents'] if result['success'] else [],
                    'num_results': result['num_results'] if result['success'] else 0
                }
                
        except Exception as e:
            error_msg = f"Error searching by source: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }
    
    def _extract_sources(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Extract unique source file names from chunks"""
        sources = set()
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            if 'file_name' in metadata:
                sources.add(metadata['file_name'])
        return sorted(list(sources))
    
    def clear_vector_store(self) -> bool:
        """Clear all documents from vector store"""
        try:
            success = self.vector_store.reset_collection()
            if success:
                logger.info("Vector store cleared successfully")
            return success
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            return False
    
    def remove_document(self, file_name: str) -> Dict[str, Any]:
        """
        Remove a specific document from the knowledge base
        
        Args:
            file_name: Name of the file to remove
            
        Returns:
            Dictionary with removal results
        """
        try:
            logger.info(f"Removing document: {file_name}")
            
            result = self.vector_store.remove_documents_by_source(file_name)
            
            if result['success']:
                logger.info(f"Document removal completed: {result['message']}")
            
            return result
            
        except Exception as e:
            error_msg = f"Error removing document {file_name}: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'file_name': file_name
            }
    
    def list_documents(self) -> Dict[str, Any]:
        """
        Get list of all documents in the knowledge base
        
        Returns:
            Dictionary with document information
        """
        try:
            return self.vector_store.get_all_sources()
        except Exception as e:
            error_msg = f"Error listing documents: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg} 