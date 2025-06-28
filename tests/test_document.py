"""
Enhanced Document Processing Test with PDF Files
Place your file in the test_documents/ folder and run this script
Default document includes Eric Zhao's resume
"""

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from document_processor import DocumentProcessor


def test_document_processing():
    """Test document processing with PDF files"""
    print("🧪 Testing Document Processing with PDF Files")
    print("="*60)
    
    processor = DocumentProcessor()
    print("✅ DocumentProcessor initialized")
    print(f"   📏 Chunk size: {processor.chunk_size}")
    print(f"   🔄 Chunk overlap: {processor.chunk_overlap}")
    print()
    
    test_folder = Path("./test_documents")
    if not test_folder.exists():
        print("❌ test_documents folder not found!")
        print("   Create it with: mkdir test_documents")
        return False
    
    pdf_files = list(test_folder.glob("*.pdf"))
    other_files = list(test_folder.glob("*.txt")) + list(test_folder.glob("*.docx"))
    all_files = pdf_files + other_files
    
    if not all_files:
        print("❌ No supported files found in test_documents/")
        print("   📋 Supported formats:", processor.get_supported_extensions())
        print("   💡 Place your resume PDF in the test_documents/ folder")
        return False
    
    print(f"📁 Found {len(all_files)} file(s) to process:")
    for file in all_files:
        print(f"   📄 {file.name} ({file.stat().st_size / 1024:.1f} KB)")
    print()
    
    results = []
    for file_path in all_files:
        print(f"🔄 Processing: {file_path.name}")
        print("-" * 40)
        
        validation = processor.validate_file(str(file_path))
        if not validation['valid']:
            print(f"❌ Validation failed: {validation['errors']}")
            continue
        
        print("✅ File validation passed")
        print(f"   📊 File info: {validation['file_info']['size_mb']:.2f} MB")
        
        result = processor.process_uploaded_file(str(file_path))
        
        if not result['success']:
            print(f"❌ Processing failed: {result['error']}")
            continue
        
        results.append(result)
        
        print("✅ Processing completed!")
        print(f"   📄 Documents loaded: {result['documents_loaded']}")
        print(f"   🧩 Chunks created: {result['chunks_created']}")
        print(f"   📝 Total characters: {result['total_characters']:,}")
        print(f"   📏 Average chunk size: {result['average_chunk_size']:.0f} chars")
        
        # Show first few chunks with preview
        chunks = result['chunks']
        print(f"\n📋 Chunk Details:")
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            print(f"   Chunk {i+1}:")
            print(f"     📝 Text preview: {chunk.text[:100]}...")
            print(f"     📊 Length: {len(chunk.text)} chars")
            print(f"     🏷️  Metadata keys: {list(chunk.metadata.keys())}")
            if 'chunk_id' in chunk.metadata:
                print(f"     🆔 Chunk ID: {chunk.metadata['chunk_id']}")
        
        if len(chunks) > 3:
            print(f"   ... and {len(chunks) - 3} more chunks")
        print()
    
    if results:
        total_chunks = sum(r['chunks_created'] for r in results)
        total_chars = sum(r['total_characters'] for r in results)
        
        print("🎉 PROCESSING SUMMARY")
        print("=" * 40)
        print(f"📄 Files processed: {len(results)}")
        print(f"🧩 Total chunks: {total_chunks}")
        print(f"📝 Total characters: {total_chars:,}")
        print(f"📏 Avg chunk size: {total_chars / total_chunks:.0f} chars")
        
        print("\nRun: python test_vectors.py")
        
        return True
    else:
        print("❌ No files were successfully processed")
        return False


if __name__ == "__main__":
    print("📋 DOCUMENT PROCESSING TEST")
    print("Place your document(s) in test_documents/ folder first!")
    print()
    
    success = test_document_processing()
    
    if success:
        print("\n🎉 Document processing test PASSED!")
        print("📄 Your document(s) are ready for vector processing")
    else:
        print("\n❌ Document processing test FAILED!")
        print("📋 Make sure to place supported files in test_documents/")
    
    exit(0 if success else 1) 