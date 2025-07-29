"""
ChromaDB Vector Store Module for RAG Q&A System
Implements vector storage and retrieval using ChromaDB's native capabilities
"""

import uuid
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding

try:
    from .config import Config
except ImportError:
    from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """
    ChromaDB vector store with native retrieval capabilities
    Using lower-level ChromaDB operations for full control
    """
    
    def __init__(self, collection_name: Optional[str] = None, persist_path: Optional[str] = None):
        """
        Initialize ChromaDB vector store
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_path: Path to persist the database
        """
        self.collection_name = collection_name or Config.CHROMA_COLLECTION_NAME
        self.persist_path = persist_path or Config.CHROMA_DB_PATH
        
        # Create persist directory if it doesn't exist
        Path(self.persist_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding model
        self.embedding_model = OpenAIEmbedding(
            model=Config.DEFAULT_EMBEDDING_MODEL,
            api_key=Config.OPENAI_API_KEY
        )
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
        
        logger.info(f"ChromaVectorStore initialized with collection '{self.collection_name}' at '{self.persist_path}'")
    
    def _get_or_create_collection(self):
        """Get existing collection or create new one"""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Retrieved existing collection '{self.collection_name}' with {collection.count()} documents")
        except NotFoundError:
            # Collection doesn't exist, create it
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "RAG Q&A System document collection"}
            )
            logger.info(f"Created new collection '{self.collection_name}'")
        
        return collection
    
    def add_nodes(self, nodes: List[TextNode]) -> Dict[str, Any]:
        """
        Add text nodes to the vector store
        
        Args:
            nodes: List of TextNode objects to add
            
        Returns:
            Dictionary with operation results
        """
        if not nodes:
            return {'success': False, 'error': 'No nodes provided'}
        
        try:
            logger.info(f"Adding {len(nodes)} nodes to vector store")
            
            # Prepare data for ChromaDB
            texts = []
            metadatas = []
            ids = []
            
            for node in nodes:
                # Generate unique ID if not present
                node_id = node.node_id if hasattr(node, 'node_id') and node.node_id else str(uuid.uuid4())
                
                texts.append(node.text)
                ids.append(node_id)
                
                # Prepare metadata (ChromaDB requires string values)
                metadata = {}
                for key, value in node.metadata.items():
                    # Convert all values to strings for ChromaDB compatibility
                    metadata[key] = str(value)
                
                metadatas.append(metadata)
            
            # Generate embeddings
            logger.info("Generating embeddings...")
            embeddings = self.embedding_model.get_text_embedding_batch(texts)
            
            # Add to ChromaDB collection
            logger.info("Adding documents to ChromaDB...")
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
            
            result = {
                'success': True,
                'nodes_added': len(nodes),
                'collection_count': self.collection.count(),
                'ids': ids
            }
            
            logger.info(f"Successfully added {len(nodes)} nodes. Collection now has {self.collection.count()} documents")
            return result
            
        except Exception as e:
            # Handle ChromaDB corruption issues
            if "compaction" in str(e).lower() or "metadata segment" in str(e).lower():
                logger.warning(f"ChromaDB corruption detected: {str(e)}")
                logger.info("Attempting to recover by resetting collection...")
                
                try:
                    # Reset the collection
                    if self.reset_collection():
                        # Retry adding nodes to the fresh collection
                        logger.info("Retrying node addition after collection reset...")
                        return self.add_nodes(nodes)
                    else:
                        error_msg = f"Failed to recover from ChromaDB corruption: {str(e)}"
                        logger.error(error_msg)
                        return {'success': False, 'error': error_msg}
                except Exception as recovery_error:
                    error_msg = f"Recovery failed: {str(recovery_error)}"
                    logger.error(error_msg)
                    return {'success': False, 'error': error_msg}
            else:
                error_msg = f"Error adding nodes to vector store: {str(e)}"
                logger.error(error_msg)
                return {'success': False, 'error': error_msg}
    
    def query(self, query_text: str, top_k: int = None, filters: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Query the vector store using ChromaDB's native retrieval
        
        Args:
            query_text: Text to search for
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            Dictionary with query results
        """
        top_k = top_k or Config.MAX_RETRIEVED_CHUNKS
        
        try:
            logger.info(f"Querying vector store: '{query_text[:50]}...' (top_k={top_k})")
            
            # Generate query embedding
            query_embedding = self.embedding_model.get_text_embedding(query_text)
            
            # Prepare where clause for filtering
            where_clause = None
            if filters:
                # Convert filters to ChromaDB format
                where_clause = {}
                for key, value in filters.items():
                    where_clause[key] = str(value)
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Process results
            retrieved_nodes = []
            for i in range(len(results['documents'][0])):
                node_data = {
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i]
                }
                retrieved_nodes.append(node_data)
            
            query_result = {
                'success': True,
                'query': query_text,
                'num_results': len(retrieved_nodes),
                'nodes': retrieved_nodes,
                'collection_size': self.collection.count()
            }
            
            logger.info(f"Query returned {len(retrieved_nodes)} results")
            return query_result
            
        except Exception as e:
            error_msg = f"Error querying vector store: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg, 'query': query_text}
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection"""
        try:
            count = self.collection.count()
            
            # Get sample documents if any exist
            sample_docs = []
            if count > 0:
                sample_results = self.collection.get(limit=3, include=['documents', 'metadatas'])
                for i in range(len(sample_results['documents'])):
                    sample_docs.append({
                        'text_preview': sample_results['documents'][i][:100] + "...",
                        'metadata': sample_results['metadatas'][i]
                    })
            
            return {
                'collection_name': self.collection_name,
                'document_count': count,
                'persist_path': self.persist_path,
                'sample_documents': sample_docs,
                'embedding_model': Config.DEFAULT_EMBEDDING_MODEL
            }
            
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {'error': str(e)}
    
    def delete_collection(self) -> bool:
        """Delete the entire collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection '{self.collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            return False
    
    def reset_collection(self) -> bool:
        """Reset the collection (delete all documents but keep collection)"""
        try:
            # Delete and recreate collection
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "RAG Q&A System document collection"}
            )
            logger.info(f"Reset collection '{self.collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Error resetting collection: {str(e)}")
            return False
    
    def search_by_metadata(self, metadata_filters: Dict[str, str], limit: int = 10) -> Dict[str, Any]:
        """
        Search documents by metadata filters only
        
        Args:
            metadata_filters: Dictionary of metadata key-value pairs to filter by
            limit: Maximum number of results to return
            
        Returns:
            Dictionary with search results
        """
        try:
            # Convert filters to ChromaDB format
            where_clause = {}
            for key, value in metadata_filters.items():
                where_clause[key] = str(value)
            
            results = self.collection.get(
                where=where_clause,
                limit=limit,
                include=['documents', 'metadatas']
            )
            
            documents = []
            for i in range(len(results['documents'])):
                documents.append({
                    'text': results['documents'][i],
                    'metadata': results['metadatas'][i]
                })
            
            return {
                'success': True,
                'num_results': len(documents),
                'documents': documents,
                'filters_used': metadata_filters
            }
            
        except Exception as e:
            error_msg = f"Error searching by metadata: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def remove_documents_by_source(self, file_name: str) -> Dict[str, Any]:
        """
        Remove all documents from a specific source file
        
        Args:
            file_name: Name of the source file to remove
            
        Returns:
            Dictionary with removal results
        """
        try:
            logger.info(f"Removing documents from source: {file_name}")
            
            # First, find all document IDs that match the file_name
            results = self.collection.get(
                where={"file_name": str(file_name)},
                include=['documents', 'metadatas']
            )
            
            if not results['ids']:
                return {
                    'success': True,
                    'message': f"No documents found for file: {file_name}",
                    'documents_removed': 0
                }
            
            # Delete the documents
            self.collection.delete(
                where={"file_name": str(file_name)}
            )
            
            removed_count = len(results['ids'])
            
            logger.info(f"Successfully removed {removed_count} documents from {file_name}")
            
            return {
                'success': True,
                'message': f"Removed {removed_count} document chunks from {file_name}",
                'documents_removed': removed_count,
                'file_name': file_name,
                'collection_count': self.collection.count()
            }
            
        except Exception as e:
            error_msg = f"Error removing documents from {file_name}: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def get_all_sources(self) -> Dict[str, Any]:
        """
        Get all unique source files in the collection
        
        Returns:
            Dictionary with list of sources and their document counts
        """
        try:
            # Get all documents with metadata
            results = self.collection.get(
                include=['metadatas']
            )
            
            # Count documents by source
            source_counts = {}
            for metadata in results['metadatas']:
                file_name = metadata.get('file_name', 'Unknown')
                source_counts[file_name] = source_counts.get(file_name, 0) + 1
            
            return {
                'success': True,
                'sources': list(source_counts.keys()),
                'source_counts': source_counts,
                'total_sources': len(source_counts),
                'total_documents': sum(source_counts.values())
            }
            
        except Exception as e:
            error_msg = f"Error getting sources: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg} 