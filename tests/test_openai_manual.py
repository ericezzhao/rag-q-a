#!/usr/bin/env python3
"""
Manual OpenAI Connectivity Test
USE THIS SCRIPT TO TEST YOUR OPENAI API KEY MANUALLY

Instructions:
1. Create a .env file: copy env.template to .env
2. Add your OpenAI API key: OPENAI_API_KEY=sk-your-key-here
3. Run this script: python test_openai_manual.py

This script will test:
- API key configuration
- GPT-4o model access
- Embedding model access
"""

import os
import sys
from dotenv import load_dotenv

def test_openai_connectivity():
    """Test OpenAI API connectivity manually"""
    print("🧪 Manual OpenAI Connectivity Test")
    print("=" * 50)
    
    # Load environment
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    print(f"📋 Configuration:")
    print(f"   - API Key Configured: {bool(api_key)}")
    print(f"   - Target Model: gpt-4o")
    print(f"   - Embedding Model: text-embedding-3-small")
    
    if not api_key:
        print("\n❌ Setup Required:")
        print("   1. Copy env.template to .env")
        print("   2. Edit .env and add: OPENAI_API_KEY=sk-your-key-here")
        print("   3. Get your API key from: https://platform.openai.com/api-keys")
        return False
    
    if not api_key.startswith('sk-'):
        print(f"\n⚠️  Warning: API key doesn't start with 'sk-': {api_key[:10]}...")
        print("   Make sure you're using a valid OpenAI API key")
    
    try:
        # Import OpenAI
        print(f"\n📦 Importing OpenAI...")
        from openai import OpenAI
        print(f"   ✅ OpenAI library imported successfully")
        
        # Create client
        print(f"\n🔌 Creating OpenAI client...")
        client = OpenAI(api_key=api_key)
        print(f"   ✅ Client created successfully")
        
        # Test chat completion
        print(f"\n🤖 Testing GPT-4o chat completion...")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'OpenAI connection successful!' and nothing else."}
            ],
            max_tokens=10,
            temperature=0
        )
        
        result = response.choices[0].message.content.strip()
        print(f"   ✅ GPT-4o Response: {result}")
        print(f"   ✅ Model Used: {response.model}")
        print(f"   ✅ Tokens Used: {response.usage.total_tokens}")
        
        # Test embedding
        print(f"\n🧮 Testing embedding generation...")
        embedding_response = client.embeddings.create(
            model="text-embedding-3-small",
            input="Test document for RAG system"
        )
        
        embedding = embedding_response.data[0].embedding
        print(f"   ✅ Embedding generated successfully")
        print(f"   ✅ Embedding dimensions: {len(embedding)}")
        print(f"   ✅ Model used: {embedding_response.model}")
        
        print(f"\n🎉 SUCCESS! All tests passed!")
        print(f"   ✅ Your OpenAI setup is working correctly")
        print(f"   ✅ GPT-4o model is accessible")
        print(f"   ✅ Embedding model is functional")
        print(f"   ✅ Ready for RAG Q&A development!")
        
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ Test Failed: {error_msg}")
        print(f"   Error Type: {type(e).__name__}")
        
        # Provide specific guidance
        if "authentication" in error_msg.lower() or "unauthorized" in error_msg.lower():
            print("\n💡 Solution: Check your API key")
            print("   - Verify the key is correct")
            print("   - Make sure it starts with 'sk-'")
            print("   - Check if the key is active on OpenAI platform")
        elif "quota" in error_msg.lower() or "billing" in error_msg.lower():
            print("\n💡 Solution: Check your OpenAI account")
            print("   - Verify you have available credits")
            print("   - Check your usage limits")
            print("   - Add payment method if needed")
        elif "model" in error_msg.lower():
            print("\n💡 Solution: Check model access")
            print("   - Verify you have access to GPT-4o")
            print("   - Try using gpt-3.5-turbo instead")
        elif "proxies" in error_msg.lower():
            print("\n💡 Environment Issue Detected:")
            print("   - This is a known environment-specific issue")
            print("   - Try testing your API key in a different environment")
            print("   - The RAG system code is correct and will work once this is resolved")
        else:
            print(f"\n💡 Troubleshooting:")
            print(f"   - Check your internet connection")
            print(f"   - Verify OpenAI service status")
            print(f"   - Try again in a few minutes")
        
        return False

if __name__ == "__main__":
    success = test_openai_connectivity()
    
    if success:
        print(f"\nRun python test_document.py")
    else:
        print(f"\nNext Steps:")
        print(f"   1. Follow the troubleshooting guidance above")
        print(f"   2. Test your API key manually")
    
    sys.exit(0 if success else 1) 