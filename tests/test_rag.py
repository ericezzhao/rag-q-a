"""
Complete RAG Pipeline Test
Ask specific questions about the content
"""

import sys
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rag_pipeline import RAGPipeline


def test_rag_pipeline():
    """Test complete RAG pipeline"""
    print("🧪 Testing Complete RAG Pipeline (on Eric Zhao's resume)")
    print("="*60)
    
    test_folder = Path("test_documents")
    if not test_folder.exists():
        test_folder = Path("tests/test_documents")
        if not test_folder.exists():
            print("❌ test_documents folder not found!")
            print("   Place your resume PDF in test_documents/ or tests/test_documents/")
            return False
        else:
            print(f"✅ Found documents in: {test_folder}")
    else:
        print(f"✅ Found documents in: {test_folder}")
    
    # Find files
    all_files = list(test_folder.glob("*.pdf")) + list(test_folder.glob("*.txt")) + list(test_folder.glob("*.docx"))
    
    if not all_files:
        print("❌ No files found in test_documents/")
        print("   Place your file there first or use default resume file")
        return False
    
    # Initialize RAG pipeline
    temp_dir = tempfile.mkdtemp()
    pipeline = RAGPipeline(
        vector_store_path=temp_dir,
        collection_name="resume_rag_test"
    )
    
    print("✅ RAG Pipeline initialized")
    print(f"   💾 Vector store path: {temp_dir}")
    print()
    
    # Test system status
    print("🔍 System Status Check")
    print("-" * 30)
    status = pipeline.get_system_status()
    
    print(f"📄 Document processor: {status['document_processor']['status']}")
    print(f"🗄️  Vector store: {status['vector_store']['status']}")
    print(f"🤖 LLM service: {status['llm_service']['status']}")
    print(f"🔄 Pipeline: {status['pipeline']['status']}")
    print()
    
    # Ingest all documents
    print("📥 DOCUMENT INGESTION")
    print("=" * 40)
    
    ingestion_results = []
    for file_path in all_files:
        print(f"🔄 Ingesting: {file_path.name}")
        
        result = pipeline.ingest_document(str(file_path))
        
        if result['success']:
            ingestion_results.append(result)
            stats = result['processing_stats']
            vector_stats = result['vector_store_stats']
            
            print(f"✅ Ingested successfully!")
            print(f"   📄 Documents: {stats['documents_loaded']}")
            print(f"   🧩 Chunks: {stats['chunks_created']}")
            print(f"   📝 Characters: {stats['total_characters']:,}")
            print(f"   ⏱️  Time: {result['ingestion_time_seconds']}s")
            print(f"   🗄️  Collection size: {vector_stats['total_collection_size']}")
        else:
            print(f"❌ Ingestion failed: {result['error']}")
        print()
    
    if not ingestion_results:
        print("❌ No documents were successfully ingested")
        return False
    
    # Resume-specific questions
    resume_questions = [
        "What is this person's name?",
        "What is their educational background?",
        "What programming languages do they know?",
        "What is their work experience?",
        "What projects have they worked on?",
        "What are their technical skills?",
        "What companies have they worked for?",
        "What degree do they have?",
        "What is their contact information?",
        "Summarize their professional experience"
    ]
    
    print("❓ Q&A TESTING")
    print("=" * 40)
    
    query_results = []
    for i, question in enumerate(resume_questions, 1):
        print(f"\n❓ Question {i}: {question}")
        print("-" * 50)
        
        # Query the pipeline
        result = pipeline.query(question, top_k=5)  # Increase top_k for better retrieval
        
        if result['success']:
            query_results.append(result)
            
            # answer
            print(f"🤖 Answer:")
            print(f"   {result['response']}")
            print()
            
            # Show context sources with detailed debugging
            print(f"📚 Sources used ({len(result['retrieved_chunks'])} chunks):")
            if not result['retrieved_chunks']:
                print("   ⚠️  NO CHUNKS RETRIEVED! This may be why the model says 'not enough information'")
            else:
                for j, chunk in enumerate(result['retrieved_chunks'], 1):
                    source = chunk['metadata'].get('file_name', 'Unknown')
                    preview = chunk['text'][:100]
                    print(f"   {j}. {source}")
                    print(f"      📝 Text: {preview}...")
            
            stats = result['pipeline_stats']
            print(f"⏱️  Query time: {stats['query_time_seconds']}s")
            print(f"🧩 Chunks retrieved: {stats['chunks_retrieved']}")
            
            # Show collection size for debugging
            vector_info = pipeline.vector_store.get_collection_info()
            print(f"🗄️  Total chunks in database: {vector_info['document_count']}")
            
        else:
            print(f"❌ Query failed: {result['error']}")
        
        print("\n" + "="*60)
    
    # Bonus interactive mode
    print("\n🗣️  INTERACTIVE Q&A MODE")
    print("=" * 40)
    print("Now you can ask custom questions about your resume!")
    print("Type 'quit' to exit")
    print()
    
    while True:
        try:
            user_question = input("❓ Your question: ").strip()
            
            if user_question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_question:
                continue
            
            print("\n🔍 Processing your question...")
            result = pipeline.query(user_question, top_k=5)
            
            if result['success']:
                print(f"\n🤖 Answer:")
                print(f"{result['response']}")
                
                print(f"\n📚 Sources ({len(result['retrieved_chunks'])} chunks):")
                if not result['retrieved_chunks']:
                    print("   ⚠️  NO RELEVANT CHUNKS FOUND!")
                    print("   💡 Try rephrasing your question or check if the document was properly ingested")
                else:
                    for i, chunk in enumerate(result['retrieved_chunks'], 1):
                        source = chunk['metadata'].get('file_name', 'Unknown')
                        print(f"   {i}. {source}")
                        print(f"      📝 {chunk['text'][:150]}...")
                
                print(f"\n⏱️  Response time: {result['pipeline_stats']['query_time_seconds']}s")
                
                # Show if response indicates lack of information
                if any(phrase in result['response'].lower() for phrase in [
                    "don't have", "not enough information", "no information", 
                    "cannot find", "unable to find", "insufficient"
                ]):
                    print("⚠️  🤖 Model indicated insufficient information!")
                    print("💡 This might be due to:")
                    print("   - Question not matching document content")
                    print("   - Document chunking issues")
                    
            else:
                print(f"❌ Error: {result['error']}")
            
            print("\n" + "-"*50)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print("\n🎉 RAG PIPELINE TEST SUMMARY")
    print("=" * 40)
    
    total_ingestion_time = sum(r['ingestion_time_seconds'] for r in ingestion_results)
    total_chunks = sum(r['processing_stats']['chunks_created'] for r in ingestion_results)
    
    if query_results:
        avg_query_time = sum(r['pipeline_stats']['query_time_seconds'] for r in query_results) / len(query_results)
    else:
        avg_query_time = 0
    
    print(f"📄 Documents processed: {len(ingestion_results)}")
    print(f"🧩 Total chunks created: {total_chunks}")
    print(f"⏱️ Total ingestion time: {total_ingestion_time:.2f}s")
    print(f"❓ Questions answered: {len(query_results)}")
    print(f"⏱️ Average query time: {avg_query_time:.2f}s")
    print(f"🤖 LLM model: GPT-4o")
    print(f"🗄️ Vector store: ChromaDB")
    
    # Cleanup
    print(f"\n🧹 Cleaning up...")
    try:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("✅ Cleanup completed")
    except Exception as e:
        print(f"⚠️  Cleanup warning: {str(e)}")
    
    return True


if __name__ == "__main__":
    print("📋 COMPLETE RAG PIPELINE TEST")
    print("Ask questions about your document(s)!")
    print()
    
    success = test_rag_pipeline()
    
    if success:
        print("\n🎉 RAG pipeline test COMPLETED!")
        print("🤖 RAG system is working perfectly!")
        print("Run streamlit run app.py")
    else:
        print("\n❌ RAG pipeline test FAILED!")
        print("📋 Check that your document(s) is in test_documents/")
    
    exit(0 if success else 1) 