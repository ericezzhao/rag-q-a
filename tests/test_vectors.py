"""
Enhanced Vector Store Test with Documents
Shows actual embeddings and vector operations
"""

import os
import sys
import tempfile
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from document_processor import DocumentProcessor
from vector_store import ChromaVectorStore


def test_vector_processing():
    """Test vector store with documents and show vectors"""
    print("🧪 Testing Vector Store with Documents")
    print("="*60)
    
    processor = DocumentProcessor()
    
    # Use temporary directory for vector store
    temp_dir = tempfile.mkdtemp()
    vector_store = ChromaVectorStore(
        collection_name="test_collection",
        persist_path=temp_dir
    )
    
    print("✅ Components initialized")
    print(f"   📄 Document processor ready")
    print(f"   🗄️  Vector store: {vector_store.collection_name}")
    print(f"   💾 Persist path: {temp_dir}")
    print()
    
    test_folder = Path("test_documents")
    if not test_folder.exists():
        print("❌ test_documents folder not found!")
        print("   Run: python test_document.py first")
        return False
    
    # Find files to process
    pdf_files = list(test_folder.glob("*.pdf"))
    other_files = list(test_folder.glob("*.txt")) + list(test_folder.glob("*.docx"))
    all_files = pdf_files + other_files
    
    if not all_files:
        print("❌ No files found in test_documents/")
        print("   Place your file there first or use default resume file")
        return False
    
    print(f"📁 Found {len(all_files)} file(s) to vectorize:")
    for file in all_files:
        print(f"   📄 {file.name}")
    print()
    
    # Process each file and add to vector store
    all_chunks = []
    for file_path in all_files:
        print(f"🔄 Processing: {file_path.name}")
        print("-" * 40)
        
        # Process document
        result = processor.process_uploaded_file(str(file_path))
        
        if not result['success']:
            print(f"❌ Failed to process {file_path.name}: {result['error']}")
            continue
        
        chunks = result['chunks']
        all_chunks.extend(chunks)
        
        print(f"✅ Processed {file_path.name}")
        print(f"   🧩 Chunks created: {len(chunks)}")
        
        # Add chunks to vector store
        print("🔄 Adding to vector store...")
        vector_result = vector_store.add_nodes(chunks)
        
        if not vector_result['success']:
            print(f"❌ Vector store failed: {vector_result['error']}")
            continue
        
        print("✅ Added to vector store")
        print(f"   🗄️  Nodes added: {vector_result['nodes_added']}")
        print(f"   📊 Collection size: {vector_result['collection_count']}")
        print()
    
    if not all_chunks:
        print("❌ No chunks were successfully processed")
        return False
    
    # Show vector store statistics
    collection_info = vector_store.get_collection_info()
    print("📊 VECTOR STORE STATISTICS")
    print("=" * 40)
    print(f"📄 Total documents: {collection_info['document_count']}")
    print(f"🧩 Total chunks: {len(all_chunks)}")
    print(f"🤖 Embedding model: {collection_info['embedding_model']}")
    print()
    
    # Test queries and show vector results
    test_queries = [
        "work experience",
        "education background", 
        "technical skills",
        "programming languages",
        "projects"
    ]
    
    print("🔍 TESTING VECTOR QUERIES")
    print("=" * 40)
    
    for query in test_queries:
        print(f"\n🔎 Query: '{query}'")
        print("-" * 30)
        
        result = vector_store.query(query, top_k=3)
        
        if not result['success']:
            print(f"❌ Query failed: {result['error']}")
            continue
        
        nodes = result['nodes']
        print(f"✅ Found {len(nodes)} results")
        
        for i, node in enumerate(nodes, 1):
            similarity_score = node['score']
            distance = node['distance']
            text_preview = node['text'][:100]
            
            print(f"   Result {i}:")
            print(f"     🎯 Similarity: {similarity_score:.4f}")
            print(f"     📏 Distance: {distance:.4f}")
            print(f"     📝 Text: {text_preview}...")
            
            # Show metadata
            metadata = node['metadata']
            if 'file_name' in metadata:
                print(f"     📄 Source: {metadata['file_name']}")
            if 'chunk_index' in metadata:
                print(f"     🧩 Chunk: {metadata['chunk_index']}")
    
    # Demonstrate vector similarity between chunks
    print(f"\n🔬 VECTOR ANALYSIS")
    print("=" * 40)
    
    if len(all_chunks) >= 2:
        # Get embeddings for first two chunks for comparison
        sample_texts = [all_chunks[0].text, all_chunks[1].text]
        
        try:
            embeddings = vector_store.embedding_model.get_text_embedding_batch(sample_texts)
            
            print(f"📊 Embedding dimensions: {len(embeddings[0])}")
            print(f"📈 First embedding sample: {embeddings[0][:5]}... (showing first 5)")
            print(f"📈 Vector magnitude: {np.linalg.norm(embeddings[0]):.4f}")
            
            # Calculate similarity between first two chunks
            if len(embeddings) >= 2:
                similarity = np.dot(embeddings[0], embeddings[1]) / (
                    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                )
                print(f"🎯 Similarity between chunks 1&2: {similarity:.4f}")
            
        except Exception as e:
            print(f"⚠️  Vector analysis failed: {str(e)}")
    
    print(f"\n🏷️  METADATA FILTERING TEST")
    print("=" * 40)
    
    file_names = set()
    for chunk in all_chunks:
        if 'file_name' in chunk.metadata:
            file_names.add(chunk.metadata['file_name'])
    
    for file_name in file_names:
        search_result = vector_store.search_by_metadata(
            {'file_name': file_name}, 
            limit=3
        )
        
        if search_result['success']:
            print(f"📄 {file_name}: {search_result['num_results']} chunks")
        else:
            print(f"❌ Search failed for {file_name}")
    
    print(f"\n🧹 Cleaning up temporary vector store...")
    try:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("✅ Cleanup completed")
    except Exception as e:
        print(f"⚠️  Cleanup warning: {str(e)}")
    
    print(f"\n🎉 VECTOR PROCESSING SUMMARY")
    print("=" * 40)
    print(f"📄 Files processed: {len(all_files)}")
    print(f"🧩 Total chunks vectorized: {len(all_chunks)}")
    print(f"🔍 Queries tested: {len(test_queries)}")
    print(f"🤖 Embedding model: OpenAI text-embedding-3-small")
    print(f"📊 Vector dimensions: 1536 (OpenAI standard)")
    
    print("\nRun: python test_rag.py")
    
    return True


if __name__ == "__main__":
    print("📋 VECTOR STORE TEST")
    print("Testing with your documents from test_documents/")
    print()
    
    success = test_vector_processing()
    
    if success:
        print("\n🎉 Vector store test PASSED!")
        print("🗄️  Your documents are successfully vectorized")
    else:
        print("\n❌ Vector store test FAILED!")
        print("📋 Make sure documents are in test_documents/ folder")
    
    exit(0 if success else 1) 