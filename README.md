# LLM Engine Bug Analysis

This repository contains scripts for collecting, analyzing, and understanding bugs in popular Large Language Model (LLM) inference engines.

## Folder Structure

```
llm_engine_bug_analysis/
├── scripts/           # All Python scripts
├── data/             # Generated data files and logs
├── cache/            # Cache directory for temporary files
├── requirements.txt  # Python dependencies
└── README.md        # This file
```

## Requirements

1. Python 3.8 or higher
2. Install dependencies: `pip install -r requirements.txt`
3. Environment variables:
   - `OPENAI_API_KEY`: Required for LLM-based analysis
   - `GITHUB_TOKEN`: Recommended for higher API rate limits

## Script Usage Sequence

The scripts should be run in the following order to perform a complete bug analysis.

**Note**: You can run the scripts from the current directory (root) using the `-m` flag or by adding the scripts directory to your Python path.

### 1. Initial Bug Collection: `collect_issues.py`

**Purpose**: Collects bug-related issues and PRs from popular LLM inference engines.

```bash
# Method 1: Run as module from root directory
python -m scripts.collect_issues

# Method 2: Run directly from scripts directory
cd scripts && python collect_issues.py

# Method 3: Run from root with PYTHONPATH
PYTHONPATH=scripts python scripts/collect_issues.py
```

**Output**: 
- `data/llm_bugs.json`: Raw collected issues and PRs
- Logs to console and `data/bug_collector.log`

**Engines analyzed**:
- llama.cpp
- vLLM
- DeepSpeed
- MLC-LLM
- TensorRT-LLM

### 2. Bug Classification: `classifier.py`

**Purpose**: Uses LLM to classify whether collected issues are actually bug-related.

```bash
# From root directory
python -m scripts.classifier --input-file data/llm_bugs.json

# Or from scripts directory
cd scripts && python classifier.py --input-file ../data/llm_bugs.json
```

**Output**:
- `data/analysis_cache.json`: Classification results with confidence scores
- Various bug symptom reports

### 3. Merge Results: `merge_bug_results.py`

**Purpose**: Merges bug collection data with classification results.

```bash
python -m scripts.merge_bug_results
```

**Input**: 
- `data/llm_bugs.json`: From step 1
- `data/analysis_cache.json`: From step 2

**Output**: 
- `data/merged_bug_results.json`: Combined results

### 4. Add PR Status: `add_pr_status.py`

**Purpose**: Enriches data with pull request status information (Open/Merged/Closed).

```bash
python -m scripts.add_pr_status
```

**Input**: `data/merged_bug_results.json`
**Output**: `data/final_bug_results.json`

### 5. Root Cause Analysis: `analyze_root_cause.py`

**Purpose**: Analyzes fixed bugs to identify root causes using OpenAI GPT analysis.

```bash
python -m scripts.analyze_root_cause
```

**Input**: `data/final_bug_results.json`
**Output**: 
- `data/final_bug_results_with_root_cause.json`: Results with root cause analysis
- `data/bug_diffs.json`: Code diff content for analysis
- Logs to `data/root_cause_analysis.log`

### 6. Complete Missing Analysis: `complete_missing_analysis.py`

**Purpose**: Identifies and completes missing root cause analyses for bugs that should have been analyzed.

```bash
# Check status of missing analyses
python -m scripts.complete_missing_analysis --status

# Complete all missing analyses
python -m scripts.complete_missing_analysis

# Test with limited number of bugs
python -m scripts.complete_missing_analysis --test
```

**Input**: `data/final_bug_results_with_root_cause.json`
**Output**: 
- `data/final_bug_results_with_root_cause_completed.json`: Complete analysis results
- `data/missing_bug_diffs.json`: Additional diff content
- Logs to `data/complete_missing_analysis.log`

## Additional Utility Scripts

### `analyze_confidence.py`
Analyzes confidence scores and classification accuracy of the bug classification process.

```bash
python -m scripts.analyze_confidence
```

### `fetch_fixed_bug_content.py`
Fetches detailed content for fixed bugs, including PR comments and review discussions.

```bash
python -m scripts.fetch_fixed_bug_content
```

### `test_setup.py`
Tests API connectivity and authentication setup.

```bash
python -m scripts.test_setup
```

### `example_usage.py`
Demonstrates how to use the LLMBugCollector class programmatically.

```bash
python -m scripts.example_usage
```

## Data Files Explanation

### Input/Output Flow
```
llm_bugs.json 
    ↓ (classifier.py)
analysis_cache.json
    ↓ (merge_bug_results.py)
merged_bug_results.json
    ↓ (add_pr_status.py) 
final_bug_results.json
    ↓ (analyze_root_cause.py)
final_bug_results_with_root_cause.json
    ↓ (complete_missing_analysis.py)
final_bug_results_with_root_cause_completed.json
```

### Key Data Files

- **`data/llm_bugs.json`**: Raw bug data from GitHub repositories
- **`data/analysis_cache.json`**: Classification results and confidence scores
- **`data/final_bug_results.json`**: Enriched bug data with PR status
- **`data/final_bug_results_with_root_cause.json`**: Bug data with root cause analysis
- **`data/bug_diffs.json`**: Code diff content for fixed bugs
- **`data/missing_bug_diffs.json`**: Additional diff content from completion step

### Log Files

All scripts generate detailed logs in the `data/` folder:
- `data/root_cause_analysis.log`
- `data/complete_missing_analysis.log`
- `data/add_pr_status.log`
- `data/merge_bug_results.log`
- `data/fetch_bug_content.log`

## Configuration

### `config.py`
Contains configuration for:
- GitHub repositories to analyze
- OpenAI API settings
- Rate limiting parameters
- File paths and caching options

### Rate Limiting

The scripts implement rate limiting to respect GitHub and OpenAI API limits:
- GitHub: 1 request per second (authenticated), 0.2/sec (unauthenticated)
- OpenAI: Configurable delay between requests

## Common Usage Patterns

### Full Analysis Pipeline

**Option 1: Use the convenience script**
```bash
./run_pipeline.sh  # Runs the complete pipeline automatically
```

**Option 2: Run manually from root directory**
```bash
python -m scripts.collect_issues           # Collect bugs
python -m scripts.classifier               # Classify bugs  
python -m scripts.merge_bug_results        # Merge results
python -m scripts.add_pr_status            # Add PR status
python -m scripts.analyze_root_cause       # Analyze root causes
python -m scripts.complete_missing_analysis # Complete analysis
```

### Resuming from Cache
Most scripts support resuming from cache if interrupted:
- Classification cache: `data/analysis_cache.json`
- Root cause cache: `cache/` directory
- Merge cache: `data/merged_results_cache.json`

### Testing with Limited Data
Many scripts support limiting the number of items processed for testing:
```bash
python -m scripts.complete_missing_analysis --test    # Limits to 10 bugs
```

## Error Handling

The scripts include comprehensive error handling:
- Network connectivity issues
- API rate limit handling  
- File I/O errors
- JSON parsing errors
- Missing environment variables

Check log files in `data/` for detailed error information.

## Monitoring Progress

All scripts provide detailed progress logging:
- Item counts and progress percentages
- Success/failure statistics
- Performance metrics
- Cache status information

Use the `--status` flag where available to check current progress without running analysis.
