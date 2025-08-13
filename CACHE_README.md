# LLM Bug Collector - Caching System

## Overview

The LLM Bug Collector now includes a robust caching system that allows you to resume analysis from where you left off, even if the script terminates unexpectedly. This saves time and API calls by avoiding re-analysis of already processed items.

## Features

### 🔄 Automatic Caching
- **Persistent Storage**: Analysis results are automatically saved to `analysis_cache.json`
- **Resume Capability**: Restart the script to continue from where it left off
- **Periodic Saving**: Cache is saved every 10 processed items (configurable)
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
# Run analysis with caching enabled
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
📦 Cache Status: Enabled (150 items cached)
   Cache file: analysis_cache.json
   Cache size: 45678 bytes

Starting bug collection and analysis...
This may take a while depending on the number of issues/PRs...
💡 Tip: Use Ctrl+C to interrupt and save progress. Cached results will be preserved.

Progress: 10 items processed, 3 bugs found (cached: 8, new: 2)
```

## Configuration

Cache settings can be modified in `config.py`:

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

### ⚡ Performance
- **Faster Restarts**: Skip already analyzed items
- **Reduced API Calls**: Save OpenAI API costs
- **Efficient Processing**: Only analyze new items

### 🔒 Reliability
- **Crash Recovery**: Resume after unexpected termination
- **Network Resilience**: Handle API timeouts gracefully
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
