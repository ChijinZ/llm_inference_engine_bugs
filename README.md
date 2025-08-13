# LLM Inference Engine Bug Collector

This tool collects and analyzes bug-related issues and pull requests from popular LLM inference engines using GitHub API and OpenAI's GPT-4 for intelligent classification.

## Supported Repositories

- **llama.cpp** (ggerganov/llama.cpp)
- **vLLM** (vllm-project/vllm)
- **DeepSpeed** (microsoft/DeepSpeed)
- **MLC-LLM** (mlc-ai/mlc-llm)
- **TensorRT-LLM** (NVIDIA/TensorRT-LLM)
- **TGI** (huggingface/text-generation-inference)

## Features

- 🔍 **Intelligent Classification**: Uses OpenAI GPT-4 to analyze issues/PRs and determine if they are bug-related
- 📊 **Comprehensive Data**: Collects detailed information including URLs, status, dates, descriptions, and fix references
- 🚀 **Rate Limiting**: Handles GitHub API rate limits automatically
- 🔄 **Fallback System**: Uses keyword-based classification when OpenAI API is unavailable
- 📈 **Summary Reports**: Provides detailed statistics and summaries
- 💾 **JSON Export**: Saves results in structured JSON format

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd llm_engine_bug_analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up API keys:

### GitHub Token (Recommended)
Create a GitHub personal access token for better rate limits:
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate a new token with `repo` and `public_repo` permissions
3. Set the environment variable:
```bash
export GITHUB_TOKEN='your_github_token_here'
```

**Note**: The tool can work without a GitHub token using unauthenticated API access, but you'll be limited to 60 requests/hour instead of 5,000 requests/hour. This may cause rate limiting issues when analyzing large repositories.

### OpenAI API Key
1. Get your OpenAI API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Set the environment variable:
```bash
export OPENAI_API_KEY='your_openai_api_key_here'
```

## Usage

### Basic Usage
Run the collector to analyze all repositories:
```bash
python collect_issues.py
```

### Output
The tool will:
1. Fetch issues and PRs from all repositories
2. Analyze each item using OpenAI GPT-4
3. Classify items as bug-related or not
4. Determine bug status (not confirmed, confirmed but not fixed, fixed)
5. Save results to `llm_bugs.json`
6. Display a summary in the console

## Data Structure

Each bug report contains:

```json
{
  "url": "GitHub URL of the issue/PR",
  "type": "issue" or "pr",
  "status": "not_confirmed", "confirmed_not_fixed", or "fixed",
  "date": "Creation date (ISO format)",
  "fix_url": "URL of the fix (if available)",
  "description": "Issue/PR description (truncated to 1000 chars)",
  "title": "Issue/PR title",
  "repository": "Repository name",
  "number": "Issue/PR number",
  "labels": ["List", "of", "labels"],
  "assignees": ["List", "of", "assignees"],
  "milestone": "Milestone name (if any)"
}
```

## Content Analysis

The tool now analyzes the complete content of issues and PRs using intelligent truncation:

### Full Content Analysis
- **Complete Analysis**: By default, the tool analyzes the full content of issues and PRs
- **Intelligent Truncation**: When content is too long for API limits, it intelligently preserves:
  - Title (always included)
  - First few sentences (main issue description)
  - Middle sections (error messages, stack traces)
  - Last sentence (resolution information)
  - Labels (always included)

### Configuration Options
- `use_full_content: True` - Analyze complete content (default)
- `use_full_content: False` - Use legacy truncation (first 2000 characters)
- `max_tokens_for_analysis: 6000` - Maximum tokens to send to OpenAI

## Bug Classification

The tool uses OpenAI GPT-4 to classify items as bug-related based on:

### Bug Indicators
- Bug reports, crashes, errors, exceptions
- Performance issues, memory leaks
- Incorrect behavior, unexpected output
- Compatibility issues
- Security vulnerabilities
- Bug fixes, patches

### Status Classification
- **not_confirmed**: Bug reported but not yet confirmed by maintainers
- **confirmed_not_fixed**: Bug confirmed but not yet resolved
- **fixed**: Bug has been fixed/resolved

## Configuration

### Customizing Repositories
Edit the `repositories` list in the `LLMBugCollector` class:

```python
self.repositories = [
    "your-org/your-repo",
    "another-org/another-repo"
]
```

### Adjusting Analysis Parameters
- **Content analysis**: Set `use_full_content: True` in `config.py` to analyze complete issue/PR content
- **Token limits**: Adjust `max_tokens_for_analysis` in `config.py` for OpenAI API limits
- **Rate limiting**: Adjust the `rate_limit_delay` in `config.py`
- **Item limits**: Change the `max_items_per_repo` in `config.py`

## Performance Considerations

- **API Rate Limits**: The tool respects GitHub API rate limits (5000 requests/hour for authenticated users)
- **OpenAI Costs**: Each issue/PR analysis costs ~$0.01-0.02 with GPT-4
- **Processing Time**: Analysis of 1000 items takes approximately 2-3 hours
- **Data Volume**: Results are limited to recent items (last 2 years) for performance

## Error Handling

The tool includes robust error handling:
- **GitHub API errors**: Logs errors and continues processing
- **OpenAI API errors**: Falls back to keyword-based classification
- **Rate limiting**: Automatically waits when approaching limits
- **JSON parsing errors**: Uses fallback classification

## Output Files

- `llm_bugs.json`: Complete results with metadata
- Console output: Real-time progress and summary statistics

## Example Output

```
=== LLM Inference Engine Bug Collection Summary ===
Total bug-related items found: 247

By Repository:
  ggerganov/llama.cpp: 89
  vllm-project/vllm: 67
  microsoft/DeepSpeed: 45
  mlc-ai/mlc-llm: 28
  NVIDIA/TensorRT-LLM: 18

By Type:
  issue: 189
  pr: 58

By Status:
  not_confirmed: 45
  confirmed_not_fixed: 78
  fixed: 124
```

## Troubleshooting

### Common Issues

1. **GitHub API Rate Limit Exceeded**
   - Ensure you're using a GitHub token
   - Wait for rate limit reset or use a token with higher limits

2. **OpenAI API Errors**
   - Check your API key is valid
   - Ensure you have sufficient credits
   - The tool will fall back to keyword-based classification

3. **Memory Issues**
   - Reduce the item limit in `fetch_github_data()`
   - Process repositories individually

### Debug Mode
Enable debug logging by modifying the logging level:
```python
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License.
