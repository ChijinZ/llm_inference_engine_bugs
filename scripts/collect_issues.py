#!/usr/bin/env python3
"""
LLM Inference Engine Bug Collector

This script collects and analyzes bug-related issues and PRs from popular LLM inference engines:
- llama.cpp
- vLLM
- DeepSpeed
- MLC-LLM
- TensorRT-LLM

It uses GitHub API to fetch issues/PRs and OpenAI API to classify them as bug-related.
"""

import os
import json
import time
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import requests
import logging
import config

# JSON Schema validation
try:
    from jsonschema import validate, ValidationError
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    ValidationError = Exception

# Import LLM clients
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI library not installed. OpenAI provider will not be available.")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: Google Generative AI library not installed. Gemini provider will not be available.")

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOGGING_CONFIG['level']), 
    format=config.LOGGING_CONFIG['format']
)
logger = logging.getLogger(__name__)

class IssueType(Enum):
    ISSUE = "issue"
    PR = "pr"

class BugStatus(Enum):
    NOT_CONFIRMED = "not_confirmed"
    CONFIRMED_NOT_FIXED = "confirmed_not_fixed"
    FIXED = "fixed"

@dataclass
class BugReport:
    """Data structure for bug reports"""
    url: str
    type: IssueType
    status: BugStatus
    date: str
    fix_url: Optional[str]
    description: str
    title: str
    repository: str
    number: int
    labels: List[str]
    assignees: List[str]
    milestone: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'url': self.url,
            'type': self.type.value,
            'status': self.status.value,
            'date': self.date,
            'fix_url': self.fix_url,
            'description': self.description,
            'title': self.title,
            'repository': self.repository,
            'number': self.number,
            'labels': self.labels,
            'assignees': self.assignees,
            'milestone': self.milestone
        }

class LLMBugCollector:
    """Main class for collecting and analyzing bug-related issues/PRs"""
    
    # JSON Schema for bug analysis response
    BUG_ANALYSIS_SCHEMA = {
        "type": "object",
        "properties": {
            "is_bug_related": {
                "type": "boolean",
                "description": "Whether the issue/PR is bug-related"
            },
            "status": {
                "type": "string",
                "enum": ["not_confirmed", "confirmed_not_fixed", "fixed"],
                "description": "Status of the bug"
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Confidence level of the classification"
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of the classification"
            }
        },
        "required": ["is_bug_related", "status", "confidence", "reasoning"],
        "additionalProperties": False
    }
    
    def __init__(self, api_key: str = None, github_token: str = None, provider: str = None):
        self.github_token = github_token
        
        # Determine provider
        if provider:
            self.provider = provider
        else:
            self.provider = config.LLM_CONFIG['provider']
        self.original_provider = self.provider  # Keep track of original provider
        
        # Error tracking for automatic fallback
        self.consecutive_errors = 0
        self.max_errors = config.LLM_CONFIG['max_gemini_errors']
        self.fallback_enabled = config.LLM_CONFIG['fallback_to_openai']
        
        # Initialize LLM client based on provider
        self.llm_client = self._initialize_llm_client(api_key)
        
        # Define repositories to analyze
        self.repositories = config.REPOSITORIES
        
        # GitHub API headers
        if github_token:
            self.headers = {
                'Authorization': f'token {github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            logger.info("Using authenticated GitHub API (5,000 requests/hour limit)")
        else:
            self.headers = {
                'Accept': 'application/vnd.github.v3+json'
            }
            logger.warning("Using unauthenticated GitHub API (60 requests/hour limit) - this may cause rate limiting issues")
        
        # Initialize cache
        self.cache = {}
        self.cache_file = config.CACHE_CONFIG['cache_file']
        self.save_interval = config.CACHE_CONFIG['save_interval']
        self.processed_count = 0
        
        # Throughput tracking
        self.start_time = time.time()
        self.last_throughput_report = time.time()
        self.throughput_report_interval = config.PERFORMANCE_CONFIG['throughput_report_interval']
        self.items_since_last_report = 0
        
        if config.CACHE_CONFIG['enabled']:
            self._load_cache()
            logger.info(f"Cache loaded: {len(self.cache)} items cached")
        else:
            logger.info("Caching disabled")
    
    def _initialize_llm_client(self, api_key: str = None):
        """Initialize the appropriate LLM client based on provider"""
        if self.provider == 'openai':
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI library not available. Please install it with: pip install openai")
            
            if not api_key:
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    raise ValueError("OpenAI API key not provided and OPENAI_API_KEY environment variable not set")
            
            client = OpenAI(api_key=api_key)
            logger.info(f"Initialized OpenAI client with model: {config.OPENAI_CONFIG['model']}")
            return client
            
        elif self.provider == 'gemini':
            if not GEMINI_AVAILABLE:
                raise ImportError("Google Generative AI library not available. Please install it with: pip install google-generativeai")
            
            if not api_key:
                api_key = os.getenv('GOOGLE_API_KEY')
                if not api_key:
                    raise ValueError("Google API key not provided and GOOGLE_API_KEY environment variable not set")
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(config.LLM_CONFIG['model'])
            logger.info(f"Initialized Gemini client with model: {config.LLM_CONFIG['model']}")
            return model
            
        else:
            raise ValueError(f"Unsupported provider: {self.provider}. Supported providers: 'openai', 'gemini'")
    
    def _stream_process_items(self, repo: str, item_type: str) -> tuple[List[BugReport], Dict]:
        """Stream process items page by page, analyzing each page immediately"""
        bugs = []
        total_items = 0
        cached_count = 0
        new_analysis_count = 0
        processed_count = 0
        page = 1
        
        url = f"https://api.github.com/repos/{repo}/{item_type}"
        per_page = config.GITHUB_CONFIG['per_page']
        request_timeout = config.GITHUB_CONFIG['request_timeout']
        max_retries = 4
        base_delay = config.GITHUB_CONFIG['base_retry_delay']
        
        params = {
            'state': 'all',
            'per_page': per_page,
            'sort': 'created',
            'direction': 'desc'
        }
        
        logger.info(f"Starting stream processing of {item_type} for {repo}")
        
        while True:
            params['page'] = page
            retry_count = 0
            success = False
            
            # Retry loop for each page
            while retry_count < max_retries and not success:
                try:
                    response = requests.get(url, headers=self.headers, params=params, timeout=request_timeout)
                    response.raise_for_status()
                    
                    items = response.json()
                    if not items:
                        logger.info(f"No more {item_type} found on page {page} for {repo}")
                        success = True
                        break
                    
                    # Process this page immediately
                    page_bugs, page_stats = self._process_page_items(items, repo, item_type)
                    bugs.extend(page_bugs)
                    cached_count += page_stats['cached']
                    new_analysis_count += page_stats['new']
                    processed_count += page_stats['processed']
                    total_items += len(items)
                    
                    logger.info(f"Page {page}: {len(items)} {item_type}, {len(page_bugs)} bugs found (cached: {page_stats['cached']}, new: {page_stats['new']})")
                    
                    # Report throughput for this page
                    self._report_throughput(len(items))
                    
                    # Check rate limiting
                    if 'X-RateLimit-Remaining' in response.headers:
                        remaining = int(response.headers['X-RateLimit-Remaining'])
                        if remaining < 10:
                            reset_time = int(response.headers['X-RateLimit-Reset'])
                            wait_time = reset_time - time.time() + 60
                            if wait_time > 0:
                                logger.warning(f"Rate limit approaching. Waiting {wait_time:.0f} seconds...")
                                time.sleep(wait_time)
                    
                    success = True
                    
                except requests.exceptions.SSLError as e:
                    retry_count += 1
                    delay = base_delay * (2 ** retry_count)
                    logger.warning(f"SSL error on page {page} for {repo} ({item_type}): {e}")
                    logger.info(f"Retrying in {delay} seconds... (attempt {retry_count}/{max_retries})")
                    time.sleep(delay)
                    
                except requests.exceptions.ConnectionError as e:
                    retry_count += 1
                    delay = base_delay * (2 ** retry_count)
                    logger.warning(f"Connection error on page {page} for {repo} ({item_type}): {e}")
                    logger.info(f"Retrying in {delay} seconds... (attempt {retry_count}/{max_retries})")
                    time.sleep(delay)
                    
                except requests.exceptions.Timeout as e:
                    retry_count += 1
                    delay = base_delay * (2 ** retry_count)
                    logger.warning(f"Timeout error on page {page} for {repo} ({item_type}): {e}")
                    logger.info(f"Retrying in {delay} seconds... (attempt {retry_count}/{max_retries})")
                    time.sleep(delay)
                    
                except requests.exceptions.HTTPError as e:
                    if response.status_code == 404:
                        logger.error(f"Repository {repo} not found or access denied")
                        return bugs, {'total': total_items, 'cached': cached_count, 'new': new_analysis_count, 'processed': processed_count}
                    elif response.status_code == 422:
                        logger.info(f"finish analysis in the {page} page")
                        success = True
                        items = []
                        break
                    elif response.status_code == 403:
                        retry_count += 1
                        delay = base_delay * (2 ** retry_count)
                        logger.warning(f"Rate limit exceeded on page {page} for {repo} ({item_type}): {e}")
                        logger.info(f"Retrying in {delay} seconds... (attempt {retry_count}/{max_retries})")
                        time.sleep(delay)
                    else:
                        logger.error(f"HTTP error {response.status_code} on page {page} for {repo} ({item_type}): {e}")
                        retry_count += 1
                        delay = base_delay * (2 ** retry_count)
                        logger.info(f"Retrying in {delay} seconds... (attempt {retry_count}/{max_retries})")
                        time.sleep(delay)
                        
                except requests.exceptions.RequestException as e:
                    retry_count += 1
                    delay = base_delay * (2 ** retry_count)
                    logger.warning(f"Request error on page {page} for {repo} ({item_type}): {e}")
                    logger.info(f"Retrying in {delay} seconds... (attempt {retry_count}/{max_retries})")
                    time.sleep(delay)
                    
                except Exception as e:
                    logger.error(f"Unexpected error on page {page} for {repo} ({item_type}): {e}")
                    retry_count += 1
                    delay = base_delay * (2 ** retry_count)
                    logger.info(f"Retrying in {delay} seconds... (attempt {retry_count}/{max_retries})")
                    time.sleep(delay)
            
            # If we've exhausted all retries for this page, log and continue to next page
            if not success:
                logger.error(f"Failed to fetch page {page} for {repo} ({item_type}) after {max_retries} attempts. Skipping to next page.")
                page += 1
                continue
            
            # If no items were returned, we've reached the end
            if not items:
                break
                
            page += 1
            
            # Save cache periodically (every 5 pages)
            if page % 5 == 0:
                self._save_cache()
                logger.debug(f"Cache saved after page {page-1}")
        
        logger.info(f"Completed streaming {total_items} {item_type} from {repo}")
        return bugs, {'total': total_items, 'cached': cached_count, 'new': new_analysis_count, 'processed': processed_count}
    
    def _process_page_items(self, items: List[Dict], repo: str, item_type: str) -> tuple[List[BugReport], Dict]:
        """Process a single page of items"""
        bugs = []
        cached_count = 0
        new_analysis_count = 0
        processed_count = 0
        
        for item in items:
            if self._should_skip_item(item):
                continue
            
            # Check cache first
            cached_analysis = self._get_cached_analysis(item, repo)
            if cached_analysis:
                analysis = cached_analysis
                cached_count += 1
                logger.debug(f"Using cached analysis for {item_type} {item.get('number', 'unknown')}")
            else:
                # Perform new analysis
                analysis = self.is_bug_related(
                    item.get('title', ''),
                    item.get('body') or '',
                    [label['name'] for label in item.get('labels', [])]
                )
                # Cache the result
                self._cache_analysis(item, repo, analysis)
                new_analysis_count += 1
            
            if analysis.get('is_bug_related', False):
                item_type_enum = IssueType.ISSUE if item_type == "issues" else IssueType.PR
                bug_report = self._create_bug_report(item, item_type_enum, repo, analysis)
                bugs.append(bug_report)
                logger.info(f"Found bug-related {item_type[:-1]}: {item['title']}")
            
            processed_count += 1
            self.processed_count += 1
            
            # Report throughput periodically
            self._report_throughput(1)
            
            # Rate limiting for LLM API (only for new analysis)
            if not cached_analysis:
                if self.provider == 'openai':
                    time.sleep(config.OPENAI_CONFIG['rate_limit_delay'])
                else:
                    time.sleep(config.LLM_CONFIG['rate_limit_delay'])
        
        return bugs, {'cached': cached_count, 'new': new_analysis_count, 'processed': processed_count}
    
    def _call_openai_api(self, system_prompt: str, content: str) -> str:
        """Call OpenAI API with JSON schema enforcement"""
        response = self.llm_client.chat.completions.create(
            model=config.OPENAI_CONFIG['model'],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ],
            temperature=config.OPENAI_CONFIG['temperature'],
            max_tokens=config.OPENAI_CONFIG['max_tokens'],
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content.strip()
    
    def _call_gemini_api(self, system_prompt: str, content: str) -> str:
        """Call Gemini API"""
        # Combine system prompt and content for Gemini
        full_prompt = f"{system_prompt}\n\n{content}"
        
        response = self.llm_client.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=config.LLM_CONFIG['temperature'],
                max_output_tokens=config.LLM_CONFIG['max_tokens']
            )
        )
        
        # Check if response was blocked or failed
        if not response.candidates:
            raise Exception("No response candidates returned from Gemini API")
        
        candidate = response.candidates[0]
        if candidate.finish_reason == 2:  # BLOCKED
            raise Exception("Gemini API blocked the response due to content safety filters")
        elif candidate.finish_reason == 3:  # STOPPED
            raise Exception("Gemini API stopped the response")
        elif candidate.finish_reason == 4:  # MAX_TOKENS
            raise Exception("Gemini API hit maximum token limit")
        elif candidate.finish_reason == 5:  # SAFETY
            raise Exception("Gemini API blocked due to safety concerns")
        
        # Check if response has content
        if not candidate.content or not candidate.content.parts:
            raise Exception("Gemini API returned empty response")
        
        return candidate.content.parts[0].text.strip()
    
    def fetch_github_data(self, repo: str, item_type: str = "issues", state: str = "all", per_page: int = None) -> List[Dict]:
        """Fetch issues or PRs from GitHub API with robust retry logic"""
        url = f"https://api.github.com/repos/{repo}/{item_type}"
        if per_page is None:
            per_page = config.GITHUB_CONFIG['per_page']
            
        params = {
            'state': state,
            'per_page': per_page,
            'sort': 'created',
            'direction': 'desc'
        }
        
        all_items = []
        page = 1
        max_retries = 2  # Increased retry limit for comprehensive fetching
        base_delay = config.GITHUB_CONFIG['base_retry_delay']
        request_timeout = config.GITHUB_CONFIG['request_timeout']
        
        while True:
            params['page'] = page
            retry_count = 0
            success = False
            
            # Retry loop for each page
            while retry_count < max_retries and not success:
                try:
                    response = requests.get(url, headers=self.headers, params=params, timeout=request_timeout)
                    response.raise_for_status()
                    
                    items = response.json()
                    if not items:
                        logger.info(f"No more items found on page {page} for {repo} ({item_type})")
                        success = True
                        break
                        
                    all_items.extend(items)
                    logger.info(f"Fetched page {page} for {repo} ({item_type}): {len(items)} items")
                    
                    # Check rate limiting
                    if 'X-RateLimit-Remaining' in response.headers:
                        remaining = int(response.headers['X-RateLimit-Remaining'])
                        if remaining < 10:
                            reset_time = int(response.headers['X-RateLimit-Reset'])
                            wait_time = reset_time - time.time() + 60
                            if wait_time > 0:
                                logger.warning(f"Rate limit approaching. Waiting {wait_time:.0f} seconds...")
                                time.sleep(wait_time)
                    
                    success = True
                    
                except requests.exceptions.SSLError as e:
                    retry_count += 1
                    delay = base_delay * (2 ** retry_count)  # Exponential backoff
                    logger.warning(f"SSL error on page {page} for {repo} ({item_type}): {e}")
                    logger.info(f"Retrying in {delay} seconds... (attempt {retry_count}/{max_retries})")
                    time.sleep(delay)
                    
                except requests.exceptions.ConnectionError as e:
                    retry_count += 1
                    delay = base_delay * (2 ** retry_count)
                    logger.warning(f"Connection error on page {page} for {repo} ({item_type}): {e}")
                    logger.info(f"Retrying in {delay} seconds... (attempt {retry_count}/{max_retries})")
                    time.sleep(delay)
                    
                except requests.exceptions.Timeout as e:
                    retry_count += 1
                    delay = base_delay * (2 ** retry_count)
                    logger.warning(f"Timeout error on page {page} for {repo} ({item_type}): {e}")
                    logger.info(f"Retrying in {delay} seconds... (attempt {retry_count}/{max_retries})")
                    time.sleep(delay)
                    
                except requests.exceptions.HTTPError as e:
                    if response.status_code == 404:
                        logger.error(f"Repository {repo} not found or access denied")
                        return all_items
                    elif response.status_code == 403:
                        retry_count += 1
                        delay = base_delay * (2 ** retry_count)
                        logger.warning(f"Rate limit exceeded on page {page} for {repo} ({item_type}): {e}")
                        logger.info(f"Retrying in {delay} seconds... (attempt {retry_count}/{max_retries})")
                        time.sleep(delay)
                    else:
                        logger.error(f"HTTP error {response.status_code} on page {page} for {repo} ({item_type}): {e}")
                        retry_count += 1
                        delay = base_delay * (2 ** retry_count)
                        logger.info(f"Retrying in {delay} seconds... (attempt {retry_count}/{max_retries})")
                        time.sleep(delay)
                        
                except requests.exceptions.RequestException as e:
                    retry_count += 1
                    delay = base_delay * (2 ** retry_count)
                    logger.warning(f"Request error on page {page} for {repo} ({item_type}): {e}")
                    logger.info(f"Retrying in {delay} seconds... (attempt {retry_count}/{max_retries})")
                    time.sleep(delay)
                    
                except Exception as e:
                    logger.error(f"Unexpected error on page {page} for {repo} ({item_type}): {e}")
                    retry_count += 1
                    delay = base_delay * (2 ** retry_count)
                    logger.info(f"Retrying in {delay} seconds... (attempt {retry_count}/{max_retries})")
                    time.sleep(delay)
            
            # If we've exhausted all retries for this page, log and continue to next page
            if not success:
                logger.error(f"Failed to fetch page {page} for {repo} ({item_type}) after {max_retries} attempts. Skipping to next page.")
                page += 1
                continue
            
            # If no items were returned, we've reached the end
            if not items:
                break
                
            page += 1
            
            # Continue fetching until no more items are available
            # No artificial limits - fetch ALL items
        
        logger.info(f"Completed fetching {len(all_items)} {item_type} from {repo}")
        
        # Validate completeness
        if len(all_items) > 0:
            # Check if we might have missed items due to pagination issues
            expected_pages = len(all_items) // per_page + (1 if len(all_items) % per_page > 0 else 0)
            logger.info(f"Fetched {len(all_items)} {item_type} across approximately {expected_pages} pages")
            
            # No artificial limits - we fetch ALL available items
        
        return all_items
    
    def _get_cache_key(self, item: Dict, repo: str) -> str:
        """Generate a unique cache key for an item"""
        item_id = item.get('id', item.get('number', 0))
        item_type = 'issue' if 'pull_request' not in item else 'pr'
        return f"{repo}:{item_type}:{item_id}"
    
    def _get_url_from_cache_key(self, cache_key: str) -> str:
        """Extract GitHub URL from cache key"""
        try:
            # Parse cache key: "repo:type:id"
            parts = cache_key.split(':')
            if len(parts) != 3:
                raise ValueError(f"Invalid cache key format: {cache_key}")
            
            repo, item_type, item_id = parts
            
            # Construct GitHub URL
            if item_type == 'issue':
                return f"https://github.com/{repo}/issues/{item_id}"
            elif item_type == 'pr':
                return f"https://github.com/{repo}/pull/{item_id}"
            else:
                raise ValueError(f"Unknown item type: {item_type}")
                
        except Exception as e:
            logger.error(f"Error extracting URL from cache key '{cache_key}': {e}")
            return ""
    
    def _get_cache_key_info(self, cache_key: str) -> Dict[str, str]:
        """Extract information from cache key"""
        try:
            parts = cache_key.split(':')
            if len(parts) != 3:
                return {"error": f"Invalid cache key format: {cache_key}"}
            
            repo, item_type, item_id = parts
            return {
                "repository": repo,
                "item_type": item_type,
                "item_id": item_id,
                "url": self._get_url_from_cache_key(cache_key)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _load_cache(self):
        """Load cache from file"""
        if not config.CACHE_CONFIG['enabled']:
            return
            
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                # Check cache version compatibility
                if cache_data.get('version') != config.CACHE_CONFIG['cache_version']:
                    logger.warning("Cache version mismatch, starting fresh cache")
                    return
                
                # Check cache age
                cache_timestamp = cache_data.get('timestamp', 0)
                max_age_seconds = config.CACHE_CONFIG['max_cache_age_days'] * 24 * 3600
                if time.time() - cache_timestamp > max_age_seconds:
                    logger.info("Cache is too old, starting fresh cache")
                    return
                
                self.cache = cache_data.get('items', {})
                logger.info(f"Cache loaded successfully: {len(self.cache)} items")
            else:
                logger.info("No existing cache file found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            self.cache = {}
    
    def _save_cache(self):
        """Save cache to file"""
        if not config.CACHE_CONFIG['enabled']:
            return
            
        try:
            # Create backup if enabled
            if config.CACHE_CONFIG['backup_cache'] and os.path.exists(self.cache_file):
                backup_file = f"{self.cache_file}.backup"
                import shutil
                shutil.copy2(self.cache_file, backup_file)
            
            cache_data = {
                'version': config.CACHE_CONFIG['cache_version'],
                'timestamp': time.time(),
                'items': self.cache
            }
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Cache saved: {len(self.cache)} items")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def _get_cached_analysis(self, item: Dict, repo: str) -> Optional[Dict[str, Any]]:
        """Get cached analysis result for an item"""
        if not config.CACHE_CONFIG['enabled']:
            return None
            
        cache_key = self._get_cache_key(item, repo)
        cached_result = self.cache.get(cache_key)
        
        if cached_result:
            logger.debug(f"Cache hit for {cache_key}")
            return cached_result
        
        return None
    
    def _cache_analysis(self, item: Dict, repo: str, analysis: Dict[str, Any]):
        """Cache analysis result for an item"""
        if not config.CACHE_CONFIG['enabled']:
            return
            
        cache_key = self._get_cache_key(item, repo)
        self.cache[cache_key] = analysis
        logger.debug(f"Cached analysis for {cache_key}")
    
    def clear_cache(self):
        """Clear the cache and remove cache file"""
        if not config.CACHE_CONFIG['enabled']:
            logger.info("Caching is disabled")
            return
            
        self.cache = {}
        try:
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
                logger.info("Cache file removed")
            if os.path.exists(f"{self.cache_file}.backup"):
                os.remove(f"{self.cache_file}.backup")
                logger.info("Cache backup file removed")
        except Exception as e:
            logger.error(f"Error removing cache files: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not config.CACHE_CONFIG['enabled']:
            return {"enabled": False}
            
        cache_size = len(self.cache)
        cache_file_size = 0
        if os.path.exists(self.cache_file):
            cache_file_size = os.path.getsize(self.cache_file)
        
        # Analyze cache contents
        issue_count = 0
        pr_count = 0
        bug_count = 0
        
        for cache_key, analysis in self.cache.items():
            if cache_key.endswith(':issue'):
                issue_count += 1
            elif cache_key.endswith(':pr'):
                pr_count += 1
            
            if analysis.get('is_bug_related', False):
                bug_count += 1
        
        return {
            "enabled": True,
            "cache_size": cache_size,
            "cache_file_size_bytes": cache_file_size,
            "cache_file": self.cache_file,
            "issues": issue_count,
            "pull_requests": pr_count,
            "bug_related": bug_count
        }
    
    def get_cached_urls(self) -> List[Dict[str, str]]:
        """Get all cached URLs with their analysis results"""
        if not config.CACHE_CONFIG['enabled']:
            return []
        
        cached_urls = []
        for cache_key, analysis in self.cache.items():
            info = self._get_cache_key_info(cache_key)
            if "error" not in info:
                info["analysis"] = analysis
                cached_urls.append(info)
        
        return cached_urls
    
    def get_bug_related_urls(self) -> List[Dict[str, str]]:
        """Get only bug-related cached URLs"""
        if not config.CACHE_CONFIG['enabled']:
            return []
        
        bug_urls = []
        for cache_key, analysis in self.cache.items():
            if analysis.get('is_bug_related', False):
                info = self._get_cache_key_info(cache_key)
                if "error" not in info:
                    info["analysis"] = analysis
                    bug_urls.append(info)
        
        return bug_urls
    
    def validate_fetch_completeness(self, repo: str, issues: List[Dict], prs: List[Dict]) -> Dict[str, Any]:
        """Validate the completeness of fetched data and provide insights"""
        total_issues = len(issues)
        total_prs = len(prs)
        total_items = total_issues + total_prs
        
        # Check for potential issues
        warnings = []
        completeness_score = 100
        
        # No item limit check - we're fetching ALL items
        # Completeness score remains high since we're not artificially limiting
        
        # Check for reasonable item counts (very low counts might indicate issues)
        if total_items < 10:
            warnings.append(f"Very low item count ({total_items}). This might indicate fetching issues.")
            completeness_score -= 30
        
        # Check item age distribution
        if total_items > 0:
            try:
                # Sample some items to check age distribution
                sample_size = min(10, total_items)
                sample_items = issues[:sample_size] + prs[:sample_size]
                
                oldest_date = None
                newest_date = None
                
                for item in sample_items:
                    created_at = datetime.fromisoformat(item['created_at'].replace('Z', '+00:00'))
                    if oldest_date is None or created_at < oldest_date:
                        oldest_date = created_at
                    if newest_date is None or created_at > newest_date:
                        newest_date = created_at
                
                if oldest_date and newest_date:
                    age_span_days = (newest_date - oldest_date).days
                    logger.info(f"Item age span: {age_span_days} days (oldest: {oldest_date.date()}, newest: {newest_date.date()})")
                    
                    # If age span is very small, we might have missed older items
                    if age_span_days < 30 and total_items > 50:
                        warnings.append(f"Small age span ({age_span_days} days) with many items. Older items might be missing.")
                        completeness_score -= 15
                        
            except Exception as e:
                logger.warning(f"Could not analyze item age distribution: {e}")
        
        return {
            "repository": repo,
            "total_issues": total_issues,
            "total_prs": total_prs,
            "total_items": total_items,
            "completeness_score": max(0, completeness_score),
            "warnings": warnings,
            "fetch_config": {
                "per_page": config.GITHUB_CONFIG['per_page'],
                "timeout": config.GITHUB_CONFIG['request_timeout']
            }
        }
    
    def is_bug_related(self, title: str, body: str, labels: List[str]) -> Dict[str, Any]:
        """Use LLM to determine if an issue/PR is bug-related and classify its status"""
        
        # Prepare the content for analysis - use full body content
        # If body is too long, we'll need to handle it in chunks or summarize
        full_content = f"Title: {title}\n\nDescription: {body}\n\nLabels: {', '.join(labels)}"
        
        # Check if content is too long for single API call
        max_tokens_for_content = config.CONTENT_CONFIG['max_tokens_for_analysis']
        estimated_tokens = len(full_content.split()) * 1.3  # Rough token estimation
        
        if not config.CONTENT_CONFIG['use_full_content']:
            # Use legacy truncation if configured
            content = f"Title: {title}\n\nDescription: {body[:config.CONTENT_CONFIG['max_description_length']]}\n\nLabels: {', '.join(labels)}"
        elif estimated_tokens > max_tokens_for_content:
            # Content is too long, use intelligent truncation
            logger.info(f"Content too long ({estimated_tokens:.0f} estimated tokens), using intelligent truncation")
            content = self._truncate_content_intelligently(title, body, labels, max_tokens_for_content)
        else:
            content = full_content
        
        system_prompt = """You are an expert at analyzing GitHub issues and pull requests to determine if they are bug-related.

Analyze the given issue/PR and return a JSON response with the following structure:
{
    "is_bug_related": boolean,
    "status": "not_confirmed" | "confirmed_not_fixed" | "fixed",
    "confidence": float (0.0-1.0),
    "reasoning": "brief explanation of your classification"
}

Bug-related indicators:
- Bug reports, crashes, errors, exceptions
- Performance issues, memory leaks
- Incorrect behavior, unexpected output
- Compatibility issues
- Security vulnerabilities
- Bug fixes, patches

Status classification:
- "not_confirmed": Bug reported but not yet confirmed by maintainers
- "confirmed_not_fixed": Bug confirmed but not yet resolved
- "fixed": Bug has been fixed/resolved

Focus on actual bugs, not feature requests, documentation updates, or general improvements."""

        try:
            if self.provider == 'openai':
                result_text = self._call_openai_api(system_prompt, content)
            elif self.provider == 'gemini':
                result_text = self._call_gemini_api(system_prompt, content)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            # Try to parse JSON response
            try:
                # Clean up the response - remove markdown code blocks if present
                cleaned_text = result_text.strip()
                if cleaned_text.startswith('```json'):
                    cleaned_text = cleaned_text[7:]  # Remove ```json
                if cleaned_text.endswith('```'):
                    cleaned_text = cleaned_text[:-3]  # Remove ```
                cleaned_text = cleaned_text.strip()
                
                result = json.loads(cleaned_text)
                self.consecutive_errors = 0  # Reset error counter on success
                return result
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse {self.provider} response as JSON: {result_text}")
                # Fallback: simple keyword-based classification
                return self._fallback_classification(title, body, labels)
                
        except Exception as e:
            logger.error(f"{self.provider.capitalize()} API error: {e}")
            
            # Track consecutive errors for automatic fallback
            self.consecutive_errors += 1
            
            # For Gemini API blocks, try with a more sanitized prompt
            if self.provider == 'gemini' and "blocked" in str(e).lower():
                logger.info("Attempting fallback with sanitized content...")
                try:
                    # Try with a simpler, more sanitized prompt
                    sanitized_content = self._sanitize_content_for_gemini(title, body, labels)
                    sanitized_prompt = """Analyze this GitHub issue/PR and return JSON:
{
    "is_bug_related": true/false,
    "status": "not_confirmed/confirmed_not_fixed/fixed",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}

Content: """ + sanitized_content
                    
                    result_text = self._call_gemini_api(sanitized_prompt, "")
                    # Parse JSON response
                    cleaned_text = result_text.strip()
                    if cleaned_text.startswith('```json'):
                        cleaned_text = cleaned_text[7:]
                    if cleaned_text.endswith('```'):
                        cleaned_text = cleaned_text[:-3]
                    cleaned_text = cleaned_text.strip()
                    
                    result = json.loads(cleaned_text)
                    self.consecutive_errors = 0  # Reset error counter on success
                    return result
                except Exception as fallback_error:
                    logger.error(f"Fallback attempt also failed: {fallback_error}")
            
            # Check if we should automatically fallback to OpenAI
            if (self.provider == 'gemini' and 
                self.fallback_enabled and 
                self.consecutive_errors >= self.max_errors):
                logger.warning(f"Too many consecutive Gemini errors ({self.consecutive_errors}). Switching to OpenAI...")
                self._switch_to_openai()
                # Retry with OpenAI
                try:
                    result_text = self._call_openai_api(system_prompt, content)
                    # Parse JSON response
                    cleaned_text = result_text.strip()
                    if cleaned_text.startswith('```json'):
                        cleaned_text = cleaned_text[7:]
                    if cleaned_text.endswith('```'):
                        cleaned_text = cleaned_text[:-3]
                    cleaned_text = cleaned_text.strip()
                    
                    result = json.loads(cleaned_text)
                    return result
                except Exception as openai_error:
                    logger.error(f"OpenAI fallback also failed: {openai_error}")
            
            return self._fallback_classification(title, body, labels)
    
    def _fallback_classification(self, title: str, body: str, labels: List[str]) -> Dict[str, Any]:
        """Fallback classification using keyword matching"""
        content = (title + ' ' + body).lower()
        
        is_bug = any(keyword in content for keyword in config.BUG_KEYWORDS)
        is_fixed = any(keyword in content for keyword in config.FIX_KEYWORDS)
        
        if not is_bug:
            status = "not_confirmed"
        elif is_fixed:
            status = "fixed"
        else:
            status = "confirmed_not_fixed"
        
        return {
            "is_bug_related": is_bug,
            "status": status,
            "confidence": 0.5,
            "reasoning": "Fallback classification using keyword matching"
        }
    
    def _truncate_content_intelligently(self, title: str, body: str, labels: List[str], max_tokens: int) -> str:
        """Intelligently truncate content while preserving the most important parts for bug analysis"""
        
        # Priority order for content preservation:
        # 1. Title (always include)
        # 2. First part of body (usually contains the main issue description)
        # 3. Labels (always include)
        # 4. Middle part if space allows
        # 5. End part if space allows
        
        title_part = f"Title: {title}\n\n"
        labels_part = f"\nLabels: {', '.join(labels)}"
        
        # Calculate available space for body
        title_tokens = len(title_part.split()) * 1.3
        labels_tokens = len(labels_part.split()) * 1.3
        available_tokens = max_tokens - title_tokens - labels_tokens - 100  # Buffer
        
        if available_tokens <= 0:
            # Very limited space, just include title and labels
            return f"{title_part}Description: [Content too long, truncated]\n{labels_part}"
        
        # Split body into sentences for better truncation
        import re
        sentences = re.split(r'[.!?]+', body)
        
        # Start with first few sentences (usually contain the main issue)
        truncated_body = ""
        current_tokens = 0
        
        # Add first 3 sentences (usually the most important)
        for i, sentence in enumerate(sentences[:3]):
            if sentence.strip():
                sentence_tokens = len(sentence.split()) * 1.3
                if current_tokens + sentence_tokens <= available_tokens * 0.6:  # Use 60% for beginning
                    truncated_body += sentence.strip() + ". "
                    current_tokens += sentence_tokens
                else:
                    break
        
        # If we have space, add some from the middle (often contains error messages, stack traces)
        if current_tokens < available_tokens * 0.8:
            middle_start = len(sentences) // 3
            middle_end = 2 * len(sentences) // 3
            
            for i in range(middle_start, min(middle_start + 2, middle_end)):
                if i < len(sentences) and sentences[i].strip():
                    sentence_tokens = len(sentences[i].split()) * 1.3
                    if current_tokens + sentence_tokens <= available_tokens * 0.8:
                        truncated_body += sentences[i].strip() + ". "
                        current_tokens += sentence_tokens
        
        # If we still have space, add the last sentence (often contains resolution info)
        if current_tokens < available_tokens and len(sentences) > 1:
            last_sentence = sentences[-1].strip()
            if last_sentence:
                sentence_tokens = len(last_sentence.split()) * 1.3
                if current_tokens + sentence_tokens <= available_tokens:
                    truncated_body += last_sentence + "."
        
        # If body is still empty, use a simple truncation
        if not truncated_body.strip():
            words = body.split()
            max_words = int(available_tokens / 1.3)
            truncated_body = " ".join(words[:max_words])
            if len(words) > max_words:
                truncated_body += " [truncated]"
        
        return f"{title_part}Description: {truncated_body}\n{labels_part}"
    
    def _sanitize_content_for_gemini(self, title: str, body: str, labels: List[str]) -> str:
        """Sanitize content to avoid Gemini API safety filters"""
        # Remove potentially problematic content
        import re
        
        # Clean title
        sanitized_title = re.sub(r'[^\w\s\-\.\(\)\[\]]', '', title)
        
        # Clean body - remove URLs, code blocks, and potentially sensitive content
        sanitized_body = body
        # Remove URLs
        sanitized_body = re.sub(r'https?://[^\s]+', '[URL]', sanitized_body)
        # Remove code blocks
        sanitized_body = re.sub(r'```[\s\S]*?```', '[CODE]', sanitized_body)
        # Remove inline code
        sanitized_body = re.sub(r'`[^`]+`', '[CODE]', sanitized_body)
        # Remove special characters that might trigger filters
        sanitized_body = re.sub(r'[^\w\s\-\.\(\)\[\],:;!?]', '', sanitized_body)
        # Limit length
        sanitized_body = sanitized_body[:500]
        
        # Clean labels
        sanitized_labels = [re.sub(r'[^\w\-]', '', label) for label in labels]
        
        return f"Title: {sanitized_title}\nDescription: {sanitized_body}\nLabels: {', '.join(sanitized_labels)}"
    
    def _report_throughput(self, items_processed: int = 1):
        """Report throughput statistics periodically"""
        current_time = time.time()
        self.items_since_last_report += items_processed
        
        # Check if it's time to report throughput
        if current_time - self.last_throughput_report >= self.throughput_report_interval:
            elapsed_since_last = current_time - self.last_throughput_report
            throughput = self.items_since_last_report / elapsed_since_last if elapsed_since_last > 0 else 0
            
            # Calculate overall throughput
            total_elapsed = current_time - self.start_time
            total_throughput = self.processed_count / total_elapsed if total_elapsed > 0 else 0
            
            # Calculate ETA if we have some data
            eta_str = ""
            if (throughput > 0 and self.processed_count > 0 and 
                config.PERFORMANCE_CONFIG['enable_eta_estimation']):
                # Estimate remaining items (rough estimate based on typical repo sizes)
                estimated_total = 10000  # Conservative estimate
                remaining_items = max(0, estimated_total - self.processed_count)
                eta_seconds = remaining_items / throughput
                eta_str = f", ETA: {self._format_duration(eta_seconds)}"
            
            logger.info(f"📊 Throughput Report: {throughput:.2f} items/sec (current), {total_throughput:.2f} items/sec (overall){eta_str}")
            
            # Reset counters
            self.last_throughput_report = current_time
            self.items_since_last_report = 0
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.0f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def _switch_to_openai(self):
        """Switch from Gemini to OpenAI provider"""
        if not self.fallback_enabled:
            return
            
        try:
            # Get OpenAI API key
            openai_key = os.getenv('OPENAI_API_KEY')
            if not openai_key:
                logger.error("Cannot switch to OpenAI: OPENAI_API_KEY not set")
                return
            
            # Check if OpenAI is available
            if not OPENAI_AVAILABLE:
                logger.error("Cannot switch to OpenAI: OpenAI library not available")
                return
            
            # Switch provider and reinitialize client
            self.provider = 'openai'
            self.llm_client = OpenAI(api_key=openai_key)
            self.consecutive_errors = 0  # Reset error counter
            
            logger.info("Successfully switched to OpenAI provider")
            
        except Exception as e:
            logger.error(f"Failed to switch to OpenAI: {e}")
            # Revert to original provider
            self.provider = self.original_provider
    
    def extract_fix_url(self, item: Dict, analysis: Dict) -> Optional[str]:
        """Extract fix URL if the bug is fixed"""
        if analysis.get('status') != 'fixed':
            return None
        
        # Check if it's a PR that was merged
        if item.get('pull_request') and item.get('merged_at'):
            return item['html_url']
        
        # Check for references to fixes in comments or body
        body = item.get('body') or ''
        if any(phrase in body.lower() for phrase in ['fixed in', 'resolved in', 'closed by']):
            # Try to extract URL from body
            import re
            url_match = re.search(r'https://github\.com/[^\s]+', body)
            if url_match:
                return url_match.group(0)
        
        return None
    
    def collect_bugs(self) -> List[BugReport]:
        """Stream-based method to collect and analyze all bug-related issues/PRs"""
        all_bugs = []
        total_processed = 0
        cached_count = 0
        new_analysis_count = 0
        fetch_validations = []
        
        try:
            for repo in self.repositories:
                logger.info(f"Processing repository: {repo}")
                
                # Stream process issues page by page
                logger.info(f"Streaming issues from {repo}...")
                issues_bugs, issues_stats = self._stream_process_items(repo, "issues")
                all_bugs.extend(issues_bugs)
                cached_count += issues_stats['cached']
                new_analysis_count += issues_stats['new']
                total_processed += issues_stats['processed']
                
                # Stream process PRs page by page
                logger.info(f"Streaming pull requests from {repo}...")
                prs_bugs, prs_stats = self._stream_process_items(repo, "pulls")
                all_bugs.extend(prs_bugs)
                cached_count += prs_stats['cached']
                new_analysis_count += prs_stats['new']
                total_processed += prs_stats['processed']
                
                # Create validation summary
                validation = {
                    "repository": repo,
                    "total_issues": issues_stats['total'],
                    "total_prs": prs_stats['total'],
                    "total_items": issues_stats['total'] + prs_stats['total'],
                    "completeness_score": 100,  # Since we're streaming all pages
                    "warnings": [],
                    "fetch_config": {
                        "per_page": config.GITHUB_CONFIG['per_page'],
                        "timeout": config.GITHUB_CONFIG['request_timeout']
                    }
                }
                fetch_validations.append(validation)
                
                logger.info(f"Repository {repo} completed:")
                logger.info(f"  - Issues: {issues_stats['total']} (bugs: {len(issues_bugs)})")
                logger.info(f"  - PRs: {prs_stats['total']} (bugs: {len(prs_bugs)})")
                logger.info(f"  - Cached: {issues_stats['cached'] + prs_stats['cached']}, New: {issues_stats['new'] + prs_stats['new']}")
            
            # Final cache save
            if config.CACHE_CONFIG['enabled']:
                self._save_cache()
                logger.info(f"Final cache saved: {len(self.cache)} items")
            
            # Final throughput report
            total_elapsed = time.time() - self.start_time
            final_throughput = total_processed / total_elapsed if total_elapsed > 0 else 0
            logger.info(f"📊 Final Throughput Report: {final_throughput:.2f} items/sec (average)")
            logger.info(f"⏱️  Total Time: {self._format_duration(total_elapsed)}")
            
            logger.info(f"Analysis complete! Processed {total_processed} items and found {len(all_bugs)} bug-related items")
            logger.info(f"Cache statistics: {cached_count} cached, {new_analysis_count} new analyses")
            return all_bugs, fetch_validations
            
        except KeyboardInterrupt:
            logger.info("Analysis interrupted by user. Saving cache...")
            if config.CACHE_CONFIG['enabled']:
                self._save_cache()
                logger.info(f"Cache saved on interruption: {len(self.cache)} items")
            raise
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            logger.info("Saving cache before exiting...")
            if config.CACHE_CONFIG['enabled']:
                self._save_cache()
                logger.info(f"Cache saved on error: {len(self.cache)} items")
            raise
    
    def _should_skip_item(self, item: Dict) -> bool:
        """Check if an item should be skipped - now processes ALL items"""
        # No filtering - process ALL items regardless of labels or age
        return False
    
    def _create_bug_report(self, item: Dict, item_type: IssueType, repo: str, analysis: Dict) -> BugReport:
        """Create a BugReport instance from GitHub item"""
        
        # Determine status
        status_str = analysis.get('status', 'not_confirmed')
        if status_str == 'not_confirmed':
            status = BugStatus.NOT_CONFIRMED
        elif status_str == 'confirmed_not_fixed':
            status = BugStatus.CONFIRMED_NOT_FIXED
        else:
            status = BugStatus.FIXED
        
        # Extract fix URL
        fix_url = self.extract_fix_url(item, analysis)
        
        return BugReport(
            url=item['html_url'],
            type=item_type,
            status=status,
            date=item['created_at'],
            fix_url=fix_url,
            description=(item.get('body') or '')[:config.CONTENT_CONFIG['max_body_length']],  # Limit description length
            title=item.get('title', ''),
            repository=repo,
            number=item.get('number', 0),
            labels=[label['name'] for label in item.get('labels', [])],
            assignees=[assignee['login'] for assignee in item.get('assignees', [])],
            milestone=item.get('milestone', {}).get('title') if item.get('milestone') else None
        )
    
    def save_results(self, bugs: List[BugReport], filename: str = None, fetch_validations: List[Dict] = None):
        """Save results to JSON file"""
        if filename is None:
            filename = config.OUTPUT_CONFIG['default_filename']
            
        results = {
            'metadata': {
                'total_bugs': len(bugs),
                'repositories_analyzed': self.repositories,
                'generated_at': datetime.now().isoformat(),
                'analysis_method': 'OpenAI GPT-4 + fallback keyword matching',
                'cache_stats': self.get_cache_stats()
            },
            'bugs': [bug.to_dict() for bug in bugs]
        }
        
        # Add fetch validation data if provided
        if fetch_validations:
            results['fetch_validation'] = {
                'repositories': fetch_validations,
                'overall_completeness': sum(v['completeness_score'] for v in fetch_validations) / len(fetch_validations) if fetch_validations else 0
            }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2 if config.OUTPUT_CONFIG['pretty_print'] else None, ensure_ascii=False)
        
        logger.info(f"Results saved to {filename}")
    
    def print_summary(self, bugs: List[BugReport]):
        """Print a comprehensive summary of collected bugs including individual repository breakdowns"""
        print(f"\n{'='*60}")
        print(f"LLM INFERENCE ENGINE BUG COLLECTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total bug-related items found: {len(bugs)}")
        
        if not bugs:
            print("No bugs found.")
            return
        
        # Group bugs by repository
        repo_bugs = {}
        for bug in bugs:
            if bug.repository not in repo_bugs:
                repo_bugs[bug.repository] = []
            repo_bugs[bug.repository].append(bug)
        
        # Print individual repository summaries
        print(f"\n{'='*60}")
        print(f"INDIVIDUAL REPOSITORY BREAKDOWNS")
        print(f"{'='*60}")
        
        for repo, repo_bug_list in sorted(repo_bugs.items()):
            print(f"\n📁 {repo}")
            print(f"   Total bugs: {len(repo_bug_list)}")
            
            # Count by type
            type_counts = {'issue': 0, 'pr': 0}
            for bug in repo_bug_list:
                type_counts[bug.type.value] += 1
            
            print(f"   📋 Issues: {type_counts['issue']}")
            print(f"   🔄 PRs: {type_counts['pr']}")
            
            # Count by status
            status_counts = {'not_confirmed': 0, 'confirmed_not_fixed': 0, 'fixed': 0}
            for bug in repo_bug_list:
                status_counts[bug.status.value] += 1
            
            print(f"   ⏳ Not confirmed: {status_counts['not_confirmed']}")
            print(f"   🔍 Confirmed but not fixed: {status_counts['confirmed_not_fixed']}")
            print(f"   ✅ Fixed: {status_counts['fixed']}")
            
            # Calculate percentages
            if len(repo_bug_list) > 0:
                fixed_pct = (status_counts['fixed'] / len(repo_bug_list)) * 100
                print(f"   📊 Fix rate: {fixed_pct:.1f}%")
        
        # Print overall summary
        print(f"\n{'='*60}")
        print(f"OVERALL SUMMARY")
        print(f"{'='*60}")
        
        # Overall counts
        type_counts = {'issue': 0, 'pr': 0}
        status_counts = {'not_confirmed': 0, 'confirmed_not_fixed': 0, 'fixed': 0}
        
        for bug in bugs:
            type_counts[bug.type.value] += 1
            status_counts[bug.status.value] += 1
        
        print(f"📊 Total Statistics:")
        print(f"   📋 Total Issues: {type_counts['issue']}")
        print(f"   🔄 Total PRs: {type_counts['pr']}")
        print(f"   ⏳ Not confirmed: {status_counts['not_confirmed']}")
        print(f"   🔍 Confirmed but not fixed: {status_counts['confirmed_not_fixed']}")
        print(f"   ✅ Fixed: {status_counts['fixed']}")
        
        # Overall percentages
        total_bugs = len(bugs)
        if total_bugs > 0:
            print(f"\n📈 Overall Percentages:")
            print(f"   Issues: {(type_counts['issue'] / total_bugs) * 100:.1f}%")
            print(f"   PRs: {(type_counts['pr'] / total_bugs) * 100:.1f}%")
            print(f"   Fix rate: {(status_counts['fixed'] / total_bugs) * 100:.1f}%")
        
        # Repository ranking
        print(f"\n🏆 Repository Ranking (by bug count):")
        repo_ranking = sorted(repo_bugs.items(), key=lambda x: len(x[1]), reverse=True)
        for i, (repo, repo_bug_list) in enumerate(repo_ranking, 1):
            print(f"   {i}. {repo}: {len(repo_bug_list)} bugs")
        
        # Status distribution across repositories
        print(f"\n📊 Status Distribution by Repository:")
        for repo, repo_bug_list in repo_ranking:
            status_counts = {'not_confirmed': 0, 'confirmed_not_fixed': 0, 'fixed': 0}
            for bug in repo_bug_list:
                status_counts[bug.status.value] += 1
            
            print(f"   {repo}:")
            print(f"     ⏳ Not confirmed: {status_counts['not_confirmed']}")
            print(f"     🔍 Confirmed but not fixed: {status_counts['confirmed_not_fixed']}")
            print(f"     ✅ Fixed: {status_counts['fixed']}")
        
        print(f"\n{'='*60}")
        print(f"Summary completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

def main():
    """Main function to run the bug collector"""
    
    # Parse command line arguments
    import sys
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == '--clear-cache':
            # Clear cache only
            print("Clearing cache...")
            # We need to create a minimal collector to access cache methods
            provider = config.LLM_CONFIG['provider']
            if provider == 'openai':
                api_key = os.getenv('OPENAI_API_KEY', 'dummy_key_for_cache_clear')
            elif provider == 'gemini':
                api_key = os.getenv('GOOGLE_API_KEY', 'dummy_key_for_cache_clear')
            else:
                api_key = 'dummy_key_for_cache_clear'
            collector = LLMBugCollector(api_key, provider=provider)
            collector.clear_cache()
            print("Cache cleared successfully!")
            return
        elif command == '--cache-stats':
            # Show cache statistics only
            print("Cache Statistics:")
            provider = config.LLM_CONFIG['provider']
            if provider == 'openai':
                api_key = os.getenv('OPENAI_API_KEY', 'dummy_key_for_cache_stats')
            elif provider == 'gemini':
                api_key = os.getenv('GOOGLE_API_KEY', 'dummy_key_for_cache_stats')
            else:
                api_key = 'dummy_key_for_cache_stats'
            collector = LLMBugCollector(api_key, provider=provider)
            stats = collector.get_cache_stats()
            if stats['enabled']:
                print(f"  Cache enabled: Yes")
                print(f"  Cached items: {stats['cache_size']}")
                print(f"  Issues: {stats['issues']}")
                print(f"  Pull Requests: {stats['pull_requests']}")
                print(f"  Bug-related: {stats['bug_related']}")
                print(f"  Cache file size: {stats['cache_file_size_bytes']} bytes")
                print(f"  Cache file: {stats['cache_file']}")
            else:
                print(f"  Cache enabled: No")
            return
        elif command == '--cached-urls':
            # Show all cached URLs
            print("Cached URLs:")
            provider = config.LLM_CONFIG['provider']
            if provider == 'openai':
                api_key = os.getenv('OPENAI_API_KEY', 'dummy_key_for_urls')
            elif provider == 'gemini':
                api_key = os.getenv('GOOGLE_API_KEY', 'dummy_key_for_urls')
            else:
                api_key = 'dummy_key_for_urls'
            collector = LLMBugCollector(api_key, provider=provider)
            urls = collector.get_cached_urls()
            print(f"  Total cached items: {len(urls)}")
            for url_info in urls[:10]:  # Show first 10
                print(f"    {url_info['url']} ({url_info['item_type']})")
            if len(urls) > 10:
                print(f"    ... and {len(urls) - 10} more")
            return
        elif command == '--bug-urls':
            # Show bug-related URLs
            print("Bug-related URLs:")
            provider = config.LLM_CONFIG['provider']
            if provider == 'openai':
                api_key = os.getenv('OPENAI_API_KEY', 'dummy_key_for_bug_urls')
            elif provider == 'gemini':
                api_key = os.getenv('GOOGLE_API_KEY', 'dummy_key_for_bug_urls')
            else:
                api_key = 'dummy_key_for_bug_urls'
            collector = LLMBugCollector(api_key, provider=provider)
            bug_urls = collector.get_bug_related_urls()
            print(f"  Bug-related items: {len(bug_urls)}")
            for url_info in bug_urls[:10]:  # Show first 10
                status = url_info['analysis'].get('status', 'unknown')
                confidence = url_info['analysis'].get('confidence', 0)
                print(f"    {url_info['url']} - {status} (confidence: {confidence})")
            if len(bug_urls) > 10:
                print(f"    ... and {len(bug_urls) - 10} more")
            return
        elif command == '--help':
            print("LLM Bug Collector - Usage:")
            print(f"  Provider: {config.LLM_CONFIG['provider']} ({config.LLM_CONFIG['model']})")
            print("  python collect_issues.py                    # Run analysis with caching")
            print("  python collect_issues.py --clear-cache      # Clear cache")
            print("  python collect_issues.py --cache-stats      # Show cache statistics")
            print("  python collect_issues.py --cached-urls      # Show all cached URLs")
            print("  python collect_issues.py --bug-urls         # Show bug-related URLs")
            print("  python collect_issues.py --help             # Show this help")
            print()
            print("Environment Variables:")
            if config.LLM_CONFIG['provider'] == 'openai':
                print("  OPENAI_API_KEY - Your OpenAI API key")
            elif config.LLM_CONFIG['provider'] == 'gemini':
                print("  GOOGLE_API_KEY - Your Google API key (get from https://makersuite.google.com/app/apikey)")
            print("  GITHUB_TOKEN - Your GitHub token (optional, for higher rate limits)")
            return
        else:
            print(f"Unknown command: {command}")
            print("Use --help for usage information")
            return
    
    # Get API keys from environment variables
    github_token = os.getenv('GITHUB_TOKEN')
    provider = config.LLM_CONFIG['provider']
    
    if provider == 'openai':
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("Error: OPENAI_API_KEY environment variable not set")
            print("Please set your OpenAI API key: export OPENAI_API_KEY='your_key_here'")
            return
    elif provider == 'gemini':
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            print("Error: GOOGLE_API_KEY environment variable not set")
            print("Please set your Google API key: export GOOGLE_API_KEY='your_key_here'")
            print("Get your API key from: https://makersuite.google.com/app/apikey")
            return
    else:
        print(f"Error: Unsupported provider '{provider}' in config")
        return
    
    if not github_token:
        print("⚠️  Warning: GITHUB_TOKEN environment variable not set")
        print("   The tool will use unauthenticated GitHub API (60 requests/hour limit)")
        print("   This may cause rate limiting issues with large repositories")
        print("   Consider setting: export GITHUB_TOKEN='your_token_here'")
        print()
    
    # Create collector and run analysis
    collector = LLMBugCollector(api_key, github_token, provider)
    
    # Show provider and cache information
    print(f"🤖 LLM Provider: {provider.capitalize()} ({config.LLM_CONFIG['model']})")
    
    cache_stats = collector.get_cache_stats()
    if cache_stats['enabled']:
        print(f"📦 Cache Status: Enabled ({cache_stats['cache_size']} items cached)")
        print(f"   Cache file: {cache_stats['cache_file']}")
        print(f"   Cache size: {cache_stats['cache_file_size_bytes']} bytes")
    else:
        print("📦 Cache Status: Disabled")
    print()
    
    print("Starting bug collection and analysis...")
    print("This may take a while depending on the number of issues/PRs...")
    print("💡 Tip: Use Ctrl+C to interrupt and save progress. Cached results will be preserved.")
    print()
    
    try:
        bugs, fetch_validations = collector.collect_bugs()
        
        # Save results
        collector.save_results(bugs, fetch_validations=fetch_validations)
        
        # Print summary
        collector.print_summary(bugs)
        
        print(f"\nDetailed results saved to data/llm_bugs.json")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user.")
        print("   Progress has been saved to cache.")
        print("   You can resume by running the script again.")
        return

if __name__ == "__main__":
    main()
