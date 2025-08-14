#!/usr/bin/env python3
"""
Configuration file for the LLM Bug Collector

This file contains all configurable parameters that can be modified
without changing the main code.
"""

# List of repositories to analyze
REPOSITORIES = [
    "ggerganov/llama.cpp",
    "vllm-project/vllm", 
    "microsoft/DeepSpeed",
    "mlc-ai/mlc-llm",
    "NVIDIA/TensorRT-LLM",
    "huggingface/text-generation-inference"
]

# GitHub API configuration
GITHUB_CONFIG = {
    'per_page': 100,  # Number of items per page (GitHub API maximum)
    'base_retry_delay': 2,  # Base delay in seconds for exponential backoff
    'request_timeout': 30,  # Request timeout in seconds
    'retry_on_ssl_error': True,  # Whether to retry on SSL errors
    'retry_on_connection_error': True,  # Whether to retry on connection errors
    'retry_on_timeout': True,  # Whether to retry on timeout errors
}

# LLM API configuration
LLM_CONFIG = {
    'provider': 'openai',  # 'openai' or 'gemini'
    'model': 'gpt-4.1-mini',  # Model to use for analysis
    'temperature': 0.1,  # Lower temperature for more consistent results
    'max_tokens': 1000,  # Maximum tokens in response
    'rate_limit_delay': 0.1,  # Delay between API calls (seconds)
    'fallback_to_openai': True,  # Whether to fallback to OpenAI if Gemini fails
    'max_gemini_errors': 5,  # Maximum consecutive Gemini errors before switching to OpenAI
}

# Performance monitoring configuration
PERFORMANCE_CONFIG = {
    'throughput_report_interval': 60,  # Report throughput every N seconds
    'enable_eta_estimation': True,  # Whether to show ETA in throughput reports
}

# OpenAI API configuration (legacy, kept for compatibility)
OPENAI_CONFIG = {
    'model': 'gpt-4.1-mini',  # Model to use for analysis
    'temperature': 0.1,  # Lower temperature for more consistent results
    'max_tokens': 1000,  # Maximum tokens in response
    'rate_limit_delay': 0.1,  # Delay between API calls (seconds)
}

# Content processing configuration
CONTENT_CONFIG = {
    'max_description_length': 2000,  # Maximum characters to send to OpenAI (legacy, now using intelligent truncation)
    'max_body_length': 1400,  # Maximum characters to store in results
    'max_tokens_for_analysis': 6000,  # Maximum tokens to send to OpenAI for analysis
    'use_full_content': True,  # Whether to analyze full content or truncate
}

# Labels to skip (items with these labels will be ignored)
# Set to empty to process ALL items without label filtering
SKIP_LABELS = set()

# Output configuration
OUTPUT_CONFIG = {
    'default_filename': 'llm_bugs.json',
    'include_metadata': True,
    'pretty_print': True,
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'log_to_file': False,
    'log_filename': 'bug_collector.log',
}

# Analysis configuration
ANALYSIS_CONFIG = {
    'use_openai': True,  # Whether to use OpenAI for classification
    'fallback_to_keywords': True,  # Whether to use keyword fallback
    'confidence_threshold': 0.5,  # Minimum confidence for bug classification
}

# Bug status mapping
BUG_STATUS_MAPPING = {
    'not_confirmed': 'Bug reported but not yet confirmed by maintainers',
    'confirmed_not_fixed': 'Bug confirmed but not yet resolved', 
    'fixed': 'Bug has been fixed/resolved'
}

# Bug indicators for keyword-based classification
BUG_KEYWORDS = [
    'bug', 'crash', 'error', 'exception', 'fix', 'issue', 'problem', 'fail',
    'broken', 'defect', 'fault', 'glitch', 'malfunction', 'outage', 'breakdown'
]

# Fix keywords for status detection
FIX_KEYWORDS = [
    'fix', 'resolved', 'closed', 'merged', 'solved', 'patched', 'corrected',
    'repaired', 'restored', 'recovered', 'resolved'
]

# File patterns to look for in fix references
FIX_URL_PATTERNS = [
    r'https://github\.com/[^\s]+',
    r'#\d+',  # Issue/PR references
    r'pull/\d+',
    r'issues/\d+'
]

# Cache configuration
CACHE_CONFIG = {
    'enabled': True,  # Whether to use caching
    'cache_file': 'analysis_cache.json',  # Cache file name
    'save_interval': 10,  # Save cache every N processed items
    'backup_cache': True,  # Create backup of cache file
    'cache_version': '1.0',  # Cache version for compatibility
    'max_cache_age_days': 30,  # Maximum age of cached results (days)
}
