"""
LLM Service Module for RAG Q&A System
Handles hosted LLM integration with OpenAI and response generation
Model: GPT-4o
"""

import logging
from typing import List, Dict, Any, Optional, Union
import json
from dataclasses import dataclass

from llama_index.llms.openai import OpenAI
from llama_index.core.base.llms.types import ChatMessage, MessageRole

try:
    from .config import Config
except ImportError:
    from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RAGContext:
    """Container for RAG context information"""
    query: str
    retrieved_chunks: List[Dict[str, Any]]
    context_text: str
    metadata: Dict[str, Any]


class LLMService:
    """
    LLM service for generating responses using hosted models
    Handles prompt construction, context integration, and response generation
    """
    
    def __init__(self, model_name: Optional[str] = None, temperature: Optional[float] = None):
        """
        Initialize LLM service
        
        Args:
            model_name: Name of the model to use
            temperature: Temperature setting for response generation
        """
        self.model_name = model_name or Config.DEFAULT_LLM_MODEL
        self.temperature = temperature or Config.LLM_TEMPERATURE
        
        # initialize model
        self.llm = OpenAI(
            model=self.model_name,
            api_key=Config.OPENAI_API_KEY,
            temperature=self.temperature,
            max_tokens=Config.MAX_RESPONSE_TOKENS
        )
        
        logger.info(f"LLMService initialized with model={self.model_name}, temperature={self.temperature}")
    
    def create_rag_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Create a comprehensive RAG prompt with context and query
        
        Args:
            query: User's question
            context_chunks: Retrieved document chunks with metadata
            
        Returns:
            Formatted prompt string
        """
        context_parts = []
        unique_sources = set()
        
        for i, chunk in enumerate(context_chunks, 1):
            chunk_text = chunk.get('text', '')
            metadata = chunk.get('metadata', {})
            
            file_name = metadata.get('file_name', f'Document_{i}')
            section = metadata.get('section', 'content')
            
            if section and section != 'content':
                source_citation = f"{file_name} ({section})"
                unique_sources.add(f"{file_name} ({section})")
            else:
                source_citation = file_name
                unique_sources.add(file_name)
            
            context_entry = f"[SOURCE: {source_citation}]\n{chunk_text}\n"
            context_parts.append(context_entry)
        
        context_text = "\n---\n".join(context_parts)
        source_list = ", ".join(sorted(unique_sources))
        
        # Improved prompt that's more helpful and less strict
        prompt = f"""You are a helpful assistant that answers questions based on the provided document context.

AVAILABLE SOURCES: {source_list}

CONTEXT INFORMATION:
{context_text}

QUESTION: {query}

INSTRUCTIONS:
- Answer the question using the information provided in the context above
- Always cite the specific document name when referencing information
- Use this format: "According to [document name]" or "According to [document name] ([section])" 
- Be thorough and extract relevant details from the context
- If you find partial information, provide what you can and indicate what might be missing
- Only say "I don't have enough information" if the context truly contains no relevant information

ANSWER:"""
        
        return prompt
    
    def generate_response(self, rag_context: RAGContext) -> Dict[str, Any]:
        """
        Generate a response using the LLM with RAG context
        
        Args:
            rag_context: RAGContext object with query and retrieved chunks
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            logger.info(f"Generating response for query: '{rag_context.query[:50]}...'")
            
            # Create RAG prompt
            prompt = self.create_rag_prompt(rag_context.query, rag_context.retrieved_chunks)
            
            # Generate response using LLM
            response = self.llm.complete(prompt)
            
            # Extract response text
            response_text = response.text.strip()
            
            # results
            result = {
                'success': True,
                'query': rag_context.query,
                'response': response_text,
                'context_used': {
                    'num_chunks': len(rag_context.retrieved_chunks),
                    'sources': self._extract_sources(rag_context.retrieved_chunks),
                    'total_context_chars': len(rag_context.context_text)
                },
                'model_info': {
                    'model': self.model_name,
                    'temperature': self.temperature,
                    'max_tokens': Config.MAX_RESPONSE_TOKENS
                },
                'metadata': rag_context.metadata
            }
            
            logger.info(f"Response generated successfully ({len(response_text)} chars)")
            return result
            
        except Exception as e:
            error_msg = f"Error generating LLM response: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'query': rag_context.query
            }
    
    def generate_chat_response(self, messages: List[Dict[str, str]], context_chunks: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Generate a chat response with optional RAG context
        
        Args:
            messages: List of chat messages [{'role': 'user/assistant', 'content': '...'}]
            context_chunks: Optional retrieved chunks for RAG
            
        Returns:
            Dictionary with chat response
        """
        try:
            # Convert messages to LlamaIndex format
            chat_messages = []
            for msg in messages:
                role = MessageRole.USER if msg['role'] == 'user' else MessageRole.ASSISTANT
                chat_messages.append(ChatMessage(role=role, content=msg['content']))
            
            # If context is provided, prepend system message with context
            if context_chunks:
                context_text = "\n\n".join([chunk.get('text', '') for chunk in context_chunks])
                system_content = f"Use the following context to help answer questions:\n\n{context_text}"
                system_msg = ChatMessage(role=MessageRole.SYSTEM, content=system_content)
                chat_messages.insert(0, system_msg)
            
            # Generate chat response
            response = self.llm.chat(chat_messages)
            
            result = {
                'success': True,
                'response': response.message.content,
                'role': 'assistant',
                'context_used': bool(context_chunks),
                'num_context_chunks': len(context_chunks) if context_chunks else 0,
                'model_info': {
                    'model': self.model_name,
                    'temperature': self.temperature
                }
            }
            
            logger.info("Chat response generated successfully")
            return result
            
        except Exception as e:
            error_msg = f"Error generating chat response: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }
    
    def _extract_sources(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Extract unique source information from chunks"""
        sources = set()
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            if 'file_name' in metadata:
                sources.add(metadata['file_name'])
        return list(sources)
    
    def test_connection(self) -> Dict[str, Any]:
        """Test LLM connection and basic functionality"""
        try:
            logger.info("Testing LLM connection...")
            
            # Simple test prompt
            test_prompt = "Hello! Please respond with 'LLM connection successful' to confirm you're working."
            response = self.llm.complete(test_prompt)
            
            result = {
                'success': True,
                'model': self.model_name,
                'response': response.text.strip(),
                'message': 'LLM connection test passed'
            }
            
            logger.info("LLM connection test successful")
            return result
            
        except Exception as e:
            error_msg = f"LLM connection test failed: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }
    # suggested utility funciton that is not used in the pipeline, can be integrated into the UI to show the context of the query
    def summarize_chunks(self, chunks: List[Dict[str, Any]], max_summary_length: int = 200) -> str:
        """
        Create a summary of retrieved chunks for better context understanding
        
        Args:
            chunks: List of retrieved chunks
            max_summary_length: Maximum length of summary
            
        Returns:
            Summary text
        """
        try:
            if not chunks:
                return "No context information available."
            
            # Combine all chunk texts
            combined_text = " ".join([chunk.get('text', '') for chunk in chunks])
            
            # Create summarization prompt
            prompt = f"""Summarize the following text in {max_summary_length} characters or less. Focus on the key information:

{combined_text}

Summary:"""
            
            response = self.llm.complete(prompt)
            summary = response.text.strip()
            
            if len(summary) > max_summary_length:
                summary = summary[:max_summary_length-3] + "..."
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating summary: {str(e)}")
            return f"Summary unavailable: {str(e)}"
    
    # LLM as a judge evaluation, auto quality assessment
    def evaluate_response_quality(self, query: str, response: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate the quality of a generated response
        
        Args:
            query: Original query
            response: Generated response
            context_chunks: Context used
            
        Returns:
            Quality evaluation metrics
        """
        try:
            evaluation_prompt = f"""Evaluate the quality of this Q&A response on a scale of 1-5 for each criterion:

QUERY: {query}

RESPONSE: {response}

CONTEXT: {len(context_chunks)} chunks provided

Rate the following (1=poor, 5=excellent):
1. Relevance to query
2. Use of provided context
3. Accuracy and factualness
4. Clarity and coherence
5. Completeness

Provide scores as: Relevance: X, Context_Use: X, Accuracy: X, Clarity: X, Completeness: X"""
            
            eval_response = self.llm.complete(evaluation_prompt)
            
            return {
                'success': True,
                'evaluation': eval_response.text.strip(),
                'query': query,
                'response_length': len(response),
                'context_chunks_used': len(context_chunks)
            }
            
        except Exception as e:
            logger.error(f"Error evaluating response: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            } 