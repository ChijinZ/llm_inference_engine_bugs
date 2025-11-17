#!/usr/bin/env python3
"""
Test script to verify the setup and dependencies for the LLM Bug Collector
"""

import sys
import os
import requests
from openai import OpenAI

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import json
        import time
        from datetime import datetime
        from typing import List, Dict, Optional, Any
        from dataclasses import dataclass, asdict
        from enum import Enum
        import requests
        from openai import OpenAI
        import logging
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_github_api(github_token=None):
    """Test GitHub API access"""
    print("\nTesting GitHub API...")
    
    if not github_token:
        github_token = os.getenv('GITHUB_TOKEN')
    
    if not github_token:
        print("⚠️  No GitHub token found. Will use unauthenticated API (60 requests/hour limit)")
        print("   This may cause rate limiting issues. Consider setting GITHUB_TOKEN environment variable.")
        return True  # Still return True since unauthenticated access is possible
    
    headers = {
        'Authorization': f'token {github_token}',
        'Accept': 'application/vnd.github.v3+json'
    }
    
    try:
        # Test with a simple API call
        response = requests.get('https://api.github.com/user', headers=headers)
        response.raise_for_status()
        
        user_data = response.json()
        print(f"✅ GitHub API working. Authenticated as: {user_data.get('login', 'Unknown')}")
        
        # Check rate limits
        rate_limit = response.headers.get('X-RateLimit-Remaining', 'Unknown')
        print(f"   Rate limit remaining: {rate_limit}")
        
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ GitHub API error: {e}")
        return False

def test_openai_api(openai_key=None):
    """Test OpenAI API access"""
    print("\nTesting OpenAI API...")
    
    if not openai_key:
        openai_key = os.getenv('OPENAI_API_KEY')
    
    if not openai_key:
        print("⚠️  No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
        return False
    
    try:
        client = OpenAI(api_key=openai_key)
        
        # Test with a simple completion
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, this is a test."}],
            max_tokens=10
        )
        
        print("✅ OpenAI API working")
        return True
    except Exception as e:
        print(f"❌ OpenAI API error: {e}")
        return False

def test_repositories():
    """Test access to target repositories"""
    print("\nTesting repository access...")
    
    repositories = [
        "ggerganov/llama.cpp",
        "vllm-project/vllm", 
        "microsoft/DeepSpeed",
        "mlc-ai/mlc-llm",
        "NVIDIA/TensorRT-LLM",
        "huggingface/text-generation-inference"
    ]
    
    github_token = os.getenv('GITHUB_TOKEN')
    headers = {}
    
    if github_token:
        headers = {
            'Authorization': f'token {github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
    
    for repo in repositories:
        try:
            response = requests.get(f'https://api.github.com/repos/{repo}', headers=headers)
            response.raise_for_status()
            
            repo_data = response.json()
            print(f"✅ {repo}: {repo_data.get('description', 'No description')}")
            
        except requests.exceptions.RequestException as e:
            print(f"❌ {repo}: {e}")

def main():
    """Run all tests"""
    print("=== LLM Bug Collector Setup Test ===\n")
    
    # Test imports
    imports_ok = test_imports()
    
    # Test GitHub API
    github_ok = test_github_api()
    
    # Test OpenAI API
    openai_ok = test_openai_api()
    
    # Test repositories
    test_repositories()
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"Imports: {'✅' if imports_ok else '❌'}")
    print(f"GitHub API: {'✅' if github_ok else '❌'}")
    print(f"OpenAI API: {'✅' if openai_ok else '❌'}")
    
    if imports_ok and openai_ok:
        print("\n🎉 All tests passed! You're ready to run the bug collector.")
        print("Run: python collect_issues.py")
        if not github_ok:
            print("⚠️  Note: GitHub token not set - will use unauthenticated API (limited to 60 requests/hour)")
    else:
        print("\n⚠️  Some tests failed. Please check the setup instructions in README.md")
        sys.exit(1)

if __name__ == "__main__":
    main()
