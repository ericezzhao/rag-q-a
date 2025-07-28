"""
Document Processing Module for RAG Q&A System
Using LlamaIndex lower-level abstractions for document loading and chunking
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader, DocxReader
from llama_index.core.schema import TextNode

try:
    from .config import Config
except ImportError:
    from config import Config

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Document processor using LlamaIndex lower-level abstractions
    Handles loading, chunking, and metadata extraction
    """
    
    def __init__(self):
        """Initialize the document processor with configuration"""
        self.chunk_size = Config.CHUNK_SIZE
        self.chunk_overlap = Config.CHUNK_OVERLAP
        
        # Initialize node parser for chunking
        self.node_parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator=" ",
        )
        
        # Initialize readers for different file types (pdf, docx)
        self.pdf_reader = PDFReader()
        self.docx_reader = DocxReader()
        
        logger.info(f"DocumentProcessor initialized with chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a single document from file path
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects
            
        Raises:
            ValueError: If file type is not supported
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        logger.info(f"Loading document: {file_path} (type: {file_extension})")
        
        try:
            if file_extension == '.pdf':
                documents = self.pdf_reader.load_data(file=file_path)
            elif file_extension == '.docx':
                documents = self.docx_reader.load_data(file=file_path)
            elif file_extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                documents = [Document(text=content)]
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            for doc in documents:
                doc.metadata.update({
                    'file_name': file_path.name,
                    'file_path': str(file_path),
                    'file_size': file_path.stat().st_size,
                    'file_type': file_extension,
                    'source': 'uploaded_document'
                })
            
            logger.info(f"Successfully loaded {len(documents)} document(s) from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise
    
    def detect_section_type(self, text: str) -> str:
        """
        Detect the general content type/section of the text chunk
        Uses simple heuristics

        Can possibly removed in the future or specialized for specific RAG purposes
        
        Args:
            text: Text content to analyze
            
        Returns:
            Basic section type or 'content' for general content
        """
        text_lower = text.lower().strip()
        
        if len(text_lower) < 50:
            return 'header'
        elif any(marker in text_lower for marker in ['table of contents', 'contents', 'index']):
            return 'table_of_contents'
        elif any(marker in text_lower for marker in ['abstract', 'summary', 'overview', 'introduction']):
            return 'introduction'
        elif any(marker in text_lower for marker in ['conclusion', 'conclusions', 'summary', 'final']):
            return 'conclusion'
        elif any(marker in text_lower for marker in ['references', 'bibliography', 'citations']):
            return 'references'
        elif any(marker in text_lower for marker in ['appendix', 'appendices', 'attachment']):
            return 'appendix'
        else:
            return 'content'
    
    def chunk_documents(self, documents: List[Document]) -> List[TextNode]:
        """
        Chunk documents into smaller pieces using LlamaIndex node parser
        
        Args:
            documents: List of Document objects to chunk
            
        Returns:
            List of TextNode objects (chunks)
        """
        logger.info(f"Chunking {len(documents)} document(s)")
        
        all_nodes = []
        
        for doc_idx, document in enumerate(documents):
            try:
                # Use node parser to create chunks
                nodes = self.node_parser.get_nodes_from_documents([document])
                
                # Add metadata to each node (doc_id, chunk_id, chunk_index, total_chunks, chunk_size, section)
                for node_idx, node in enumerate(nodes):
                    section_type = self.detect_section_type(node.text)
                    
                    node.metadata.update({
                        'doc_id': f"doc_{doc_idx}",
                        'chunk_id': f"doc_{doc_idx}_chunk_{node_idx}",
                        'chunk_index': node_idx,
                        'total_chunks': len(nodes),
                        'chunk_size': len(node.text),
                        'section': section_type,
                        **document.metadata
                    })
                
                all_nodes.extend(nodes)
                logger.info(f"Document {doc_idx} chunked into {len(nodes)} pieces")
                
            except Exception as e:
                logger.error(f"Error chunking document {doc_idx}: {str(e)}")
                raise
        
        logger.info(f"Total chunks created: {len(all_nodes)}")
        return all_nodes
    
    def process_uploaded_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process an uploaded file end-to-end: load and chunk
        
        Args:
            file_path: Path to the uploaded file
            
        Returns:
            Dictionary with processing results
        """
        try:
            documents = self.load_document(file_path)
            
            chunks = self.chunk_documents(documents)
            
            total_chars = sum(len(chunk.text) for chunk in chunks)
            avg_chunk_size = total_chars / len(chunks) if chunks else 0
            
            result = {
                'success': True,
                'file_path': file_path,
                'documents_loaded': len(documents),
                'chunks_created': len(chunks),
                'total_characters': total_chars,
                'average_chunk_size': avg_chunk_size,
                'chunks': chunks,
                'metadata': documents[0].metadata if documents else {}
            }
            
            logger.info(f"Successfully processed {file_path}: {len(chunks)} chunks created")
            return result
            
        except Exception as e:
            error_msg = f"Error processing file {file_path}: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'file_path': file_path
            }
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions"""
        return ['.pdf', '.docx', '.txt']
    
    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate if a file can be processed
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with validation results
        """
        file_path = Path(file_path)
        
        validation = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'file_info': {}
        }
        
        if not file_path.exists():
            validation['valid'] = False
            validation['errors'].append(f"File not found: {file_path}")
            return validation
        
        extension = file_path.suffix.lower()
        if extension not in self.get_supported_extensions():
            validation['valid'] = False
            validation['errors'].append(f"Unsupported file type: {extension}")
        
        file_size = file_path.stat().st_size
        max_size = Config.MAX_UPLOAD_SIZE_MB * 1024 * 1024  # Convert MB to bytes
        
        if file_size > max_size:
            validation['valid'] = False
            validation['errors'].append(f"File too large: {file_size / (1024*1024):.1f}MB (max: {Config.MAX_UPLOAD_SIZE_MB}MB)")
        
        if file_size == 0:
            validation['valid'] = False
            validation['errors'].append("File is empty")
        
        validation['file_info'] = {
            'name': file_path.name,
            'size_bytes': file_size,
            'size_mb': file_size / (1024 * 1024),
            'extension': extension,
            'absolute_path': str(file_path.absolute())
        }
        
        return validation 