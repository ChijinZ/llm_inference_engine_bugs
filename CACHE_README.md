# LLM Bug Collector - Multi-Provider Support & Caching System

## Overview

The LLM Bug Collector now supports multiple LLM providers (OpenAI and Google Gemini) and includes a robust caching system with **streaming processing** that allows you to resume analysis from where you left off, even if the script terminates unexpectedly. This saves time and API calls by avoiding re-analysis of already processed items.

### 🔄 Streaming Architecture

The system now processes data as a **stream**:
1. **Fetch Page** → **Process Page** → **Cache Results** → **Repeat**
2. Each page is analyzed immediately after fetching
3. Cache is saved every 5 pages for safety
4. Memory usage is optimized (no large data accumulation)

## 🤖 LLM Provider Support

### Supported Providers
- **OpenAI**: GPT-4, GPT-4o, GPT-3.5-turbo
- **Google Gemini**: gemini-2.0-flash-exp, gemini-1.5-flash, gemini-1.5-pro

### Default Configuration
The system is configured to use **Google Gemini** as the default provider:
```python
LLM_CONFIG = {
    'provider': 'gemini',  # 'openai' or 'gemini'
    'model': 'gemini-2.0-flash-exp',  # Default model
    'temperature': 0.1,
    'max_tokens': 500,
    'rate_limit_delay': 0.1,
}
```

## Features

### 🔄 Streaming Processing
- **Page-by-Page Processing**: Each page is fetched and processed immediately
- **Real-time Analysis**: LLM analysis happens as soon as data is received
- **Memory Efficient**: No accumulation of large datasets in memory
- **Immediate Caching**: Results are cached immediately after analysis

### 🔄 Automatic Caching
- **Persistent Storage**: Analysis results are automatically saved to `analysis_cache.json`
- **Resume Capability**: Restart the script to continue from where it left off
- **Periodic Saving**: Cache is saved every 5 pages (configurable)
- **Backup Protection**: Automatic backup of cache file before overwriting

### 🛡️ Robust Error Handling
- **Graceful Interruption**: Use Ctrl+C to safely stop the script
- **Automatic Recovery**: Cache is saved on interruption or error
- **Version Compatibility**: Cache version checking prevents corruption

### 📊 Cache Management
- **Statistics**: View cache size and status
- **Manual Clearing**: Clear cache when needed
- **Age Management**: Automatic cache expiration (30 days default)

## Usage

### Basic Usage
```bash
# Run analysis with Gemini (default)
python collect_issues.py

# Run analysis with OpenAI
# First, set provider in config.py: LLM_CONFIG['provider'] = 'openai'
python collect_issues.py
```

### Cache Management Commands
```bash
# View cache statistics
python collect_issues.py --cache-stats

# Clear cache
python collect_issues.py --clear-cache

# Show help
python collect_issues.py --help
```

### Example Output
```
🤖 LLM Provider: OpenAI (gpt-4o)
📦 Cache Status: Enabled (150 items cached)
   Cache file: analysis_cache.json
   Cache size: 45678 bytes

Starting bug collection and analysis...
This may take a while depending on the number of issues/PRs...
💡 Tip: Use Ctrl+C to interrupt and save progress. Cached results will be preserved.

Streaming issues from ggerganov/llama.cpp...
Page 1: 100 issues, 3 bugs found (cached: 0, new: 3)
Page 2: 100 issues, 1 bugs found (cached: 0, new: 1)
Page 3: 100 issues, 2 bugs found (cached: 0, new: 2)
📊 Throughput Report: 2.5 items/sec (current), 2.3 items/sec (overall), ETA: 45m
...
Repository ggerganov/llama.cpp completed:
  - Issues: 5000 (bugs: 45)
  - PRs: 2000 (bugs: 23)
  - Cached: 0, New: 68
📊 Final Throughput Report: 2.4 items/sec (average)
⏱️  Total Time: 52m
```

## Configuration

### LLM Provider Settings
```python
LLM_CONFIG = {
    'provider': 'gemini',              # 'openai' or 'gemini'
    'model': 'gemini-2.0-flash-exp',   # Model name
    'temperature': 0.1,                # Response randomness
    'max_tokens': 500,                 # Max response length
    'rate_limit_delay': 0.1,           # Delay between calls
}
```

### Cache Settings
```python
CACHE_CONFIG = {
    'enabled': True,                    # Enable/disable caching
    'cache_file': 'analysis_cache.json', # Cache file name
    'save_interval': 10,                # Save every N items
    'backup_cache': True,               # Create backup files
    'cache_version': '1.0',             # Cache version
    'max_cache_age_days': 30,           # Cache expiration
}
```

## Cache File Structure

The cache file (`analysis_cache.json`) contains:

```json
{
  "version": "1.0",
  "timestamp": 1640995200.0,
  "items": {
    "ggerganov/llama.cpp:issue:123": {
      "is_bug_related": true,
      "status": "fixed",
      "confidence": 0.85,
      "reasoning": "Bug report with fix"
    },
    "ggerganov/llama.cpp:pr:456": {
      "is_bug_related": false,
      "status": "not_confirmed",
      "confidence": 0.2,
      "reasoning": "Feature request"
    }
  }
}
```

## Cache Keys

Cache keys are generated using the format:
```
{repository}:{item_type}:{item_id}
```

Examples:
- `ggerganov/llama.cpp:issue:123`
- `vllm-project/vllm:pr:456`

## Benefits

### 🤖 Multi-Provider Support
- **Flexible Choice**: Use OpenAI or Google Gemini
- **Cost Optimization**: Choose the most cost-effective provider
- **Reliability**: Fallback options if one provider is unavailable

### ⚡ Performance
- **Streaming Processing**: Page-by-page analysis for better memory management
- **Faster Restarts**: Skip already analyzed items
- **Reduced API Calls**: Save API costs
- **Efficient Processing**: Only analyze new items
- **Real-time Progress**: See results as they're processed
- **Throughput Monitoring**: Track items per second with ETA estimates

### 🔒 Reliability
- **Crash Recovery**: Resume after unexpected termination
- **Network Resilience**: Handle API timeouts gracefully
- **Automatic Fallback**: Switches from Gemini to OpenAI if too many errors occur
- **Content Sanitization**: Handles Gemini API safety filters with sanitized prompts
- **JSON Schema Validation**: Ensures LLM responses match expected format
- **Data Protection**: Automatic backups prevent data loss

### 💰 Cost Savings
- **API Efficiency**: Avoid redundant OpenAI calls
- **Time Savings**: Skip re-analysis of processed items
- **Resource Optimization**: Reduce computational overhead

## Best Practices

### 🚀 For Large Repositories
1. **Start Early**: Begin analysis during off-peak hours
2. **Monitor Progress**: Check cache stats regularly
3. **Resume Strategy**: Use Ctrl+C to pause and resume later

### 🔧 Maintenance
1. **Regular Cleanup**: Clear old cache files periodically
2. **Version Updates**: Clear cache when updating script versions
3. **Storage Management**: Monitor cache file size

### 🛠️ Troubleshooting
1. **Cache Corruption**: Use `--clear-cache` to reset
2. **Version Mismatch**: Cache will auto-reset for new versions
3. **Storage Issues**: Check disk space for cache files

### 📊 Performance Monitoring
1. **Throughput Reports**: Automatically shown every 60 seconds
2. **ETA Estimation**: Based on current processing speed
3. **Real-time Metrics**: Current vs overall throughput
4. **Final Summary**: Total time and average throughput

## Setup & Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API Keys
```bash
# For OpenAI (default)
export OPENAI_API_KEY="your_openai_api_key_here"

# For Gemini (optional)
export GOOGLE_API_KEY="your_google_api_key_here"

# For GitHub (optional, for higher rate limits)
export GITHUB_TOKEN="your_github_token_here"
```

### 3. Get API Keys
- **Google Gemini**: https://makersuite.google.com/app/apikey
- **OpenAI**: https://platform.openai.com/api-keys
- **GitHub**: https://github.com/settings/tokens

## Testing

Run the cache test script to verify functionality:

```bash
python test_cache.py
```

This will test:
- Cache key generation
- Storage and retrieval
- File persistence
- Cache clearing

## Migration from Previous Versions

If you have existing results without caching:
1. The script will work normally
2. New analyses will be cached automatically
3. No data migration required

## Technical Details

### Cache Invalidation
- **Version Changes**: Automatic invalidation on version mismatch
- **Age Limits**: Configurable expiration (default: 30 days)
- **Manual Clearing**: User-initiated cache clearing

### Performance Impact
- **Memory Usage**: Minimal overhead (~1KB per cached item)
- **Disk Usage**: JSON format, human-readable
- **Load Time**: Fast cache loading on startup

### Error Handling
- **File Corruption**: Graceful fallback to fresh cache
- **Permission Issues**: Logged warnings, continued operation
- **Disk Full**: Automatic cache saving disabled, analysis continues
- **JSON Validation**: Schema validation ensures response quality

### JSON Schema Validation
- **Response Format**: Enforces structured JSON responses from LLMs
- **Field Validation**: Ensures all required fields are present
- **Type Checking**: Validates data types (boolean, string, number)
- **Enum Validation**: Restricts status values to valid options
- **Fallback Handling**: Uses keyword-based classification if validation fails
