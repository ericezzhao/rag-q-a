"""
Automation API for n8n Integration
Provides endpoints for automated document processing
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import os
import logging
from pathlib import Path
from typing import List, Dict, Any
import shutil

# Import your RAG pipeline
from src.rag_pipeline import RAGPipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Automation API", description="API for automated document processing")

# Initialize RAG pipeline
rag_pipeline = RAGPipeline()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "RAG Automation API is running", "status": "healthy"}

@app.get("/status")
async def get_status():
    """Get system status"""
    try:
        status = rag_pipeline.get_system_status()
        return JSONResponse(content=status)
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting status: {str(e)}")

@app.post("/process-document")
async def process_document(file_path: str):
    """
    Process a single document and add it to the vector database
    
    Args:
        file_path: Path to the document file (relative to monitored folder)
    """
    try:
        # Convert to absolute path if needed
        if not os.path.isabs(file_path):
            # Assume it's relative to the monitored folder
            monitored_folder = Path("monitored-folder")
            file_path = str(monitored_folder / file_path)
        
        logger.info(f"Processing document: {file_path}")
        
        # Process the document
        result = rag_pipeline.ingest_document(file_path)
        
        if result['success']:
            logger.info(f"Successfully processed {file_path}")
            return JSONResponse(content={
                "success": True,
                "message": f"Document processed successfully",
                "file_path": file_path,
                "stats": result
            })
        else:
            logger.error(f"Failed to process {file_path}: {result['error']}")
            raise HTTPException(status_code=400, detail=result['error'])
            
    except Exception as e:
        logger.error(f"Error processing document {file_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/process-monitored-folder")
async def process_monitored_folder():
    """
    Process all new documents in the monitored folder
    """
    try:
        monitored_folder = Path("monitored-folder")
        
        if not monitored_folder.exists():
            return JSONResponse(content={
                "success": True,
                "message": "Monitored folder does not exist",
                "processed_files": [],
                "skipped_files": [],
                "errors": []
            })
        
        # Get list of all files in monitored folder
        supported_extensions = rag_pipeline.document_processor.get_supported_extensions()
        files_to_process = []
        
        for file_path in monitored_folder.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                files_to_process.append(file_path)
        
        # Get list of already processed files from vector database
        existing_docs_result = rag_pipeline.list_documents()
        existing_files = []
        
        if existing_docs_result.get('success', False):
            # The API returns 'sources' not 'documents'
            if 'sources' in existing_docs_result:
                existing_files = existing_docs_result['sources']
            elif 'documents' in existing_docs_result:
                existing_files = [doc.get('file_name', '') for doc in existing_docs_result['documents']]
        
        logger.info(f"Found {len(existing_files)} existing files in vector database: {existing_files}")
        
        # Find files that exist in DB but not in folder (deleted files)
        current_files = [f.name for f in files_to_process]
        deleted_files = [f for f in existing_files if f not in current_files]
        
        # Remove deleted files from vector database
        removed_files = []
        for deleted_file in deleted_files:
            try:
                logger.info(f"Removing deleted file from database: {deleted_file}")
                result = rag_pipeline.remove_document(deleted_file)
                if result.get('success', False):
                    removed_files.append({
                        "file_name": deleted_file,
                        "success": True,
                        "message": result.get('message', 'Removed successfully')
                    })
                    logger.info(f"Successfully removed {deleted_file} from database")
                else:
                    logger.error(f"Failed to remove {deleted_file}: {result.get('error', 'Unknown error')}")
            except Exception as e:
                error_msg = f"Error removing {deleted_file}: {str(e)}"
                logger.error(error_msg)
        
        if not files_to_process:
            return JSONResponse(content={
                "success": True,
                "message": f"No supported files found in monitored folder. Removed {len(removed_files)} deleted files.",
                "processed_files": [],
                "skipped_files": [],
                "removed_files": removed_files,
                "errors": []
            })
        
        logger.info(f"Found {len(files_to_process)} files to check")
        
        processed_files = []
        skipped_files = []
        errors = []
        
        for file_path in files_to_process:
            file_name = file_path.name
            
            # Check if file was already processed
            if file_name in existing_files:
                logger.info(f"Skipping already processed file: {file_name}")
                skipped_files.append({
                    "file_path": str(file_path),
                    "file_name": file_name,
                    "reason": "Already processed"
                })
                continue
            
            try:
                logger.info(f"Processing new file: {file_name}")
                result = rag_pipeline.ingest_document(str(file_path))
                
                if result['success']:
                    processed_files.append({
                        "file_path": str(file_path),
                        "file_name": file_name,
                        "success": True,
                        "stats": result
                    })
                    logger.info(f"Successfully processed {file_name}")
                else:
                    errors.append({
                        "file_path": str(file_path),
                        "file_name": file_name,
                        "error": result['error']
                    })
                    logger.error(f"Failed to process {file_name}: {result['error']}")
                    
            except Exception as e:
                error_msg = f"Error processing {file_name}: {str(e)}"
                errors.append({
                    "file_path": str(file_path),
                    "file_name": file_name,
                    "error": error_msg
                })
                logger.error(error_msg)
        
        return JSONResponse(content={
            "success": True,
            "message": f"Processed {len(processed_files)} new files, skipped {len(skipped_files)} existing files, removed {len(removed_files)} deleted files, {len(errors)} errors",
            "processed_files": processed_files,
            "skipped_files": skipped_files,
            "removed_files": removed_files,
            "errors": errors,
            "total_files_checked": len(files_to_process)
        })
        
    except Exception as e:
        logger.error(f"Error processing monitored folder: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing monitored folder: {str(e)}")

@app.get("/documents")
async def list_documents():
    """List all documents in the vector database"""
    try:
        result = rag_pipeline.list_documents()
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@app.post("/query")
async def query_documents(request: dict):
    """Query the RAG system with a question"""
    try:
        question = request.get("query")
        if not question:
            raise HTTPException(status_code=400, detail="Question is required")
        
        logger.info(f"Processing query: {question}")
        
        result = rag_pipeline.query(question)
        
        if result['success']:
            logger.info(f"Query successful, retrieved {len(result.get('retrieved_chunks', []))} chunks")
            return JSONResponse(content={
                "success": True,
                "question": question,
                "response": result['response'],
                "sources": result.get('sources', []),
                "stats": result.get('pipeline_stats', {}),
                "retrieved_chunks": result.get('retrieved_chunks', [])
            })
        else:
            logger.error(f"Query failed: {result.get('error', 'Unknown error')}")
            raise HTTPException(status_code=400, detail=result.get('error', 'Query failed'))
            
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 