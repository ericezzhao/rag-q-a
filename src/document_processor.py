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
import pandas as pd

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
        
        # CSV and Excel processing will use pandas directly
        
        logger.info(f"DocumentProcessor initialized with chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")
    
    def load_document(self, file_path: str, original_filename: Optional[str] = None) -> List[Document]:
        """
        Load a single document from file path
        
        Args:
            file_path: Path to the document file
            original_filename: Original filename to use in metadata (instead of temp file name)
            
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
                # Try different encodings to handle various text file formats
                encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                content = None
                
                for encoding in encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            content = f.read()
                        logger.info(f"Successfully read text file with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                
                if content is None:
                    # If all encodings fail, try reading as binary and decode with error handling
                    try:
                        with open(file_path, 'rb') as f:
                            binary_content = f.read()
                        content = binary_content.decode('utf-8', errors='replace')
                        logger.warning(f"Used error handling for text file encoding")
                    except Exception as e:
                        raise ValueError(f"Unable to read text file with any encoding: {str(e)}")
                
                documents = [Document(text=content)]
            elif file_extension == '.csv':
                documents = self._process_csv_file(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                documents = self._process_excel_file(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            for doc in documents:
                # Use original filename if provided, otherwise use the actual file path name
                display_name = original_filename if original_filename else file_path.name
                
                doc.metadata.update({
                    'file_name': display_name,
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
                
                # Add metadata to each node (doc_id, chunk_id, chunk_index, total_chunks, chunk_size)
                for node_idx, node in enumerate(nodes):
                    node.metadata.update({
                        'doc_id': f"doc_{doc_idx}",
                        'chunk_id': f"doc_{doc_idx}_chunk_{node_idx}",
                        'chunk_index': node_idx,
                        'total_chunks': len(nodes),
                        'chunk_size': len(node.text),
                        **document.metadata
                    })
                
                all_nodes.extend(nodes)
                logger.info(f"Document {doc_idx} chunked into {len(nodes)} pieces")
                
            except Exception as e:
                logger.error(f"Error chunking document {doc_idx}: {str(e)}")
                raise
        
        logger.info(f"Total chunks created: {len(all_nodes)}")
        return all_nodes
    
    def process_uploaded_file(self, file_path: str, original_filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Process an uploaded file end-to-end: load and chunk
        
        Args:
            file_path: Path to the uploaded file
            original_filename: Original filename to use in metadata (instead of temp file name)
            
        Returns:
            Dictionary with processing results
        """
        try:
            documents = self.load_document(file_path, original_filename=original_filename)
            
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
        return ['.pdf', '.docx', '.txt', '.csv', '.xlsx', '.xls']
    
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
    
    def _process_csv_file(self, file_path: Path) -> List[Document]:
        """
        Process CSV file and convert to document format
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            List of Document objects
        """
        try:
            # Read CSV file with encoding handling
            try:
                df = pd.read_csv(file_path)
            except UnicodeDecodeError:
                # Try different encodings for CSV files
                encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                df = None
                
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        logger.info(f"Successfully read CSV file with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                
                if df is None:
                    raise ValueError("Unable to read CSV file with any encoding")
            
            # Convert DataFrame to text representation
            text_content = self._dataframe_to_text(df, "CSV")
            
            # Create document
            document = Document(text=text_content)
            
            # Add CSV-specific metadata
            document.metadata.update({
                'file_type': 'csv',
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'data_types': {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()}
            })
            
            logger.info(f"Successfully processed CSV file: {len(df)} rows, {len(df.columns)} columns")
            return [document]
            
        except Exception as e:
            error_msg = f"Error processing CSV file {file_path}: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _process_excel_file(self, file_path: Path) -> List[Document]:
        """
        Process Excel file and convert to document format
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            List of Document objects
        """
        try:
            # Read Excel file
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            
            if not sheet_names:
                raise ValueError("Excel file contains no sheets")
            
            documents = []
            
            for sheet_name in sheet_names:
                try:
                    # Read each sheet
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    
                    # Skip empty sheets
                    if df.empty:
                        logger.info(f"Skipping empty sheet: {sheet_name}")
                        continue
                        
                except Exception as sheet_error:
                    logger.warning(f"Error reading sheet '{sheet_name}': {str(sheet_error)}")
                    continue
                
                # Convert DataFrame to text representation
                text_content = self._dataframe_to_text(df, f"Excel Sheet: {sheet_name}")
                
                # Create document for each sheet
                document = Document(text=text_content)
                
                # Add Excel-specific metadata
                document.metadata.update({
                    'file_type': 'excel',
                    'sheet_name': sheet_name,
                    'rows': len(df),
                    'columns': len(df.columns),
                    'column_names': list(df.columns),
                    'data_types': {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()},
                    'total_sheets': len(sheet_names)
                })
                
                documents.append(document)
            
            if not documents:
                raise ValueError("No valid data found in Excel file")
            
            logger.info(f"Successfully processed Excel file: {len(sheet_names)} sheets, {len(documents)} documents created")
            return documents
            
        except Exception as e:
            error_msg = f"Error processing Excel file {file_path}: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _dataframe_to_text(self, df: pd.DataFrame, source_type: str) -> str:
        """
        Convert pandas DataFrame to structured text representation
        
        Args:
            df: Pandas DataFrame
            source_type: Type of source (CSV, Excel Sheet, etc.)
            
        Returns:
            Structured text representation
        """
        # Handle empty DataFrame
        if df.empty:
            return f"{source_type} contains no data."
        
        # Get basic info
        rows, cols = df.shape
        column_names = list(df.columns)
        
        # Start building text content
        text_parts = [
            f"{source_type} Data Summary:",
            f"- Total Rows: {rows}",
            f"- Total Columns: {cols}",
            f"- Column Names: {', '.join(column_names)}",
            "",
            "Data Preview:"
        ]
        
        # Add column headers
        header_row = " | ".join(str(col) for col in column_names)
        text_parts.append(header_row)
        text_parts.append("-" * len(header_row))
        
        # Add data rows (limit to first 20 rows to avoid huge documents)
        max_preview_rows = min(20, rows)
        for i in range(max_preview_rows):
            row_data = []
            for col in column_names:
                value = df.iloc[i][col]
                # Handle NaN values and long strings
                if pd.isna(value):
                    row_data.append("N/A")
                else:
                    str_value = str(value)
                    # Truncate long values
                    if len(str_value) > 50:
                        str_value = str_value[:47] + "..."
                    row_data.append(str_value)
            
            row_text = " | ".join(row_data)
            text_parts.append(row_text)
        
        # Add summary if there are more rows
        if rows > max_preview_rows:
            text_parts.append(f"... and {rows - max_preview_rows} more rows")
        
        # Add data types information
        text_parts.append("")
        text_parts.append("Column Data Types:")
        for col, dtype in df.dtypes.items():
            text_parts.append(f"- {col}: {dtype}")
        
        # Add basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            text_parts.append("")
            text_parts.append("Numeric Column Statistics:")
            for col in numeric_cols:
                stats = df[col].describe()
                text_parts.append(f"- {col}: min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}")
        
        return "\n".join(text_parts) 