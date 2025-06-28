"""
Configuration Management for RAG Q&A System
Handles environment variables, API keys, and application settings
"""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration class for the RAG Q&A System"""
    
    # API Key
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # ChromaDB Configuration
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "rag_documents")
    
    # Application Configuration
    APP_TITLE: str = os.getenv("APP_TITLE", "RAG Q&A System")
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
    
    # Default LLM Settings
    DEFAULT_LLM_PROVIDER: str = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gpt-4o")
    DEFAULT_EMBEDDING_MODEL: str = os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-3-small")
    
    # RAG Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    MAX_RETRIEVED_CHUNKS: int = int(os.getenv("MAX_RETRIEVED_CHUNKS", "10"))
    
    # LLM Configuration
    DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    MAX_RESPONSE_TOKENS: int = int(os.getenv("MAX_RESPONSE_TOKENS", "1500"))
    
    @classmethod
    def validate_api_keys(cls) -> tuple[bool, list[str]]:
        """
        Validate that at least one API key is configured
        Returns: (is_valid, missing_keys)
        """
        missing_keys = []
        
        if not cls.OPENAI_API_KEY:
            missing_keys.append("OPENAI_API_KEY")
        
        # At least one API key should be configured
        is_valid = cls.OPENAI_API_KEY
        
        return is_valid, missing_keys
    
    @classmethod
    def get_available_providers(cls) -> list[str]:
        """Get list of available LLM providers based on configured API keys"""
        providers = []
        
        if cls.OPENAI_API_KEY:
            providers.append("openai")
        
        return providers
    
    @classmethod
    def display_debug_info(cls) -> dict:
        """Get configuration info for debugging (without exposing API keys)"""
        return {
            "app_title": cls.APP_TITLE,
            "debug_mode": cls.DEBUG_MODE,
            "max_upload_size_mb": cls.MAX_UPLOAD_SIZE_MB,
            "chroma_db_path": cls.CHROMA_DB_PATH,
            "chroma_collection_name": cls.CHROMA_COLLECTION_NAME,
            "default_llm_provider": cls.DEFAULT_LLM_PROVIDER,
            "default_model": cls.DEFAULT_MODEL,
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP,
            "max_retrieved_chunks": cls.MAX_RETRIEVED_CHUNKS,
            "available_providers": cls.get_available_providers(),
            "openai_key_configured": bool(cls.OPENAI_API_KEY),
        } 