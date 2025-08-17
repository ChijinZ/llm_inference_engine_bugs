#!/usr/bin/env python3
"""
Script to merge bug analysis results from multiple sources:
- llm_bugs.json: contains title, description, status
- analysis_cache.json: contains is_bug_related, reasoning, confidence
- URL to cache key mapping logic from collect_issues.py
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from urllib.parse import urlparse
from common import GitHubAPIClient
import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler('merge_bug_results.log'),
                        logging.StreamHandler(sys.stdout)
                    ])
logger = logging.getLogger(__name__)


class BugResultsMerger:
    """Merge bug analysis results from multiple data sources"""

    def __init__(self, llm_bugs_file: str, analysis_cache_file: str,
                 output_file: str):
        self.llm_bugs_file = llm_bugs_file
        self.analysis_cache_file = analysis_cache_file
        self.output_file = output_file

        # Initialize GitHub API client
        self.github_client = GitHubAPIClient()

        # Load data
        self.llm_bugs_data = self.load_json_file(llm_bugs_file)
        self.analysis_cache_data = self.load_json_file(analysis_cache_file)

        logger.info(
            f"Loaded {len(self.llm_bugs_data.get('bugs', []))} bugs from {llm_bugs_file}"
        )
        logger.info(
            f"Loaded {len(self.analysis_cache_data.get('items', {}))} cache items from {analysis_cache_file}"
        )

    def load_json_file(self, file_path: str) -> Dict[str, Any]:
        """Load JSON data from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return {}

    def get_github_item_data(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Fetch GitHub item data to generate proper cache key
        Based on logic from collect_issues.py _get_cache_key function
        """
        try:
            parsed_url = urlparse(url)
            if parsed_url.hostname != 'github.com':
                return None

            # Parse URL path: /owner/repo/issues|pull/number
            path_parts = parsed_url.path.strip('/').split('/')
            if len(path_parts) < 4:
                return None

            owner, repo, item_type, item_number = path_parts[0], path_parts[
                1], path_parts[2], path_parts[3]

            # Determine API endpoint
            if item_type == 'issues':
                api_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{item_number}"
            elif item_type == 'pull':
                api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{item_number}"
            else:
                return None

            # Fetch item data from GitHub API using the client's api_get method
            response = self.github_client.api_get(api_url)
            if response and response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to fetch GitHub data for {url}")
                return None

        except Exception as e:
            logger.debug(f"Error fetching GitHub data for '{url}': {e}")
            return None

    def generate_cache_key(self, item_data: Dict[str, Any], repo: str) -> str:
        """
        Generate cache key using the same logic as collect_issues.py _get_cache_key
        """
        repo_name_map = {
            "deepspeedai/DeepSpeed":
            "microsoft/DeepSpeed",
            "ggml-org/llama.cpp":
            "ggerganov/llama.cpp",
            "vllm-project/vllm":
            "vllm-project/vllm",
            "mlc-ai/mlc-llm":
            "mlc-ai/mlc-llm",
            "NVIDIA/TensorRT-LLM":
            "NVIDIA/TensorRT-LLM",
            "huggingface/text-generation-inference":
            "huggingface/text-generation-inference"
        }

        repo = repo_name_map.get(repo)

        item_id = item_data.get('id', item_data.get('number', 0))
        item_type = 'issue' if 'pull_request' not in item_data else 'pr'
        return f"{repo}:{item_type}:{item_id}"

    def url_to_cache_key(self, url: str) -> Optional[str]:
        """
        Convert GitHub URL to cache key format by fetching item data first
        """
        try:
            parsed_url = urlparse(url)
            if parsed_url.hostname != 'github.com':
                return None

            # Parse URL to get repo
            path_parts = parsed_url.path.strip('/').split('/')
            if len(path_parts) < 2:
                return None

            owner, repo = path_parts[0], path_parts[1]
            repo_full = f"{owner}/{repo}"

            # Fetch GitHub item data
            item_data = self.get_github_item_data(url)
            if not item_data:
                return None

            # Generate cache key using the same logic as collect_issues.py
            cache_key = self.generate_cache_key(item_data, repo_full)
            return cache_key

        except Exception as e:
            logger.debug(f"Error converting URL to cache key '{url}': {e}")
            return None

    def get_cache_data(self, cache_key: str) -> Dict[str, Any]:
        """Get analysis data from cache for a given cache key"""
        cache_items = self.analysis_cache_data.get('items', {})
        cache_data = cache_items.get(cache_key, {})

        return {
            'is_bug_related': cache_data.get('is_bug_related', None),
            'reasoning_for_bug_related': cache_data.get('reasoning', ''),
            'confidence_for_bug_related': cache_data.get('confidence', None)
        }

    def merge_bug_data(self, bug: Dict[str, Any]) -> Dict[str, Any]:
        """Merge data from llm_bugs.json and analysis_cache.json for a single bug"""
        url = bug.get('url', '')

        # Get cache key from URL (this will fetch GitHub data if needed)
        cache_key = self.url_to_cache_key(url)

        if cache_key:
            logger.debug(f"Generated cache key '{cache_key}' for URL '{url}'")
        else:
            logger.warning(f"Could not generate cache key for URL '{url}'")

        # Get analysis data from cache
        cache_data = self.get_cache_data(cache_key) if cache_key else {
            'is_bug_related': None,
            'reasoning_for_bug_related': '',
            'confidence_for_bug_related': None
        }

        # Merge all required fields
        merged_result = {
            'is_bug_related': cache_data['is_bug_related'],
            'reasoning_for_bug_related':
            cache_data['reasoning_for_bug_related'],
            'confidence_for_bug_related':
            cache_data['confidence_for_bug_related'],
            'title': bug.get('title', ''),
            'description': bug.get('description', ''),
            'status': bug.get('status', '')
        }

        return merged_result

    def process_all_bugs(self) -> Dict[str, Dict[str, Any]]:
        """Process all bugs and create merged results with caching"""
        merged_results = {}
        bugs = self.llm_bugs_data.get('bugs', [])
        cache_file = "merged_results_cache.json"

        # Try to load existing cache
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_results = json.load(f)
                    merged_results.update(cached_results)
                    logger.info(
                        f"Loaded {len(cached_results)} cached results from {cache_file}"
                    )
        except Exception as e:
            logger.warning(f"Could not load cache file: {e}")

        logger.info(f"Processing {len(bugs)} bugs...")
        logger.info(f"Already have {len(merged_results)} results cached")

        # Track statistics
        successful_cache_keys = 0
        failed_cache_keys = 0
        processed_count = len(merged_results)
        cache_interval = 10  # Save cache every 10 processed bugs

        for i, bug in enumerate(bugs, 1):
            url = bug.get('url', '')
            if not url:
                logger.warning(f"Bug #{i} has no URL, skipping")
                continue

            # Skip if already processed
            if url in merged_results:
                logger.debug(f"Skipping already processed bug #{i}: {url}")
                continue

            try:
                # Merge bug data
                merged_data = self.merge_bug_data(bug)
                merged_results[url] = merged_data
                processed_count += 1

                # Track cache key success
                cache_key = self.url_to_cache_key(url)
                if cache_key:
                    successful_cache_keys += 1
                else:
                    failed_cache_keys += 1

            except Exception as e:
                logger.error(f"Error processing bug #{i} ({url}): {e}")
                # Still add the bug with basic data
                merged_results[url] = {
                    'is_bug_related': None,
                    'reasoning_for_bug_related': '',
                    'confidence_for_bug_related': None,
                    'title': bug.get('title', ''),
                    'description': bug.get('description', ''),
                    'status': bug.get('status', '')
                }
                processed_count += 1

            # Save cache periodically
            if processed_count % cache_interval == 0:
                try:
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump(merged_results,
                                  f,
                                  indent=2,
                                  ensure_ascii=False)
                    logger.info(
                        f"Saved cache with {len(merged_results)} results to {cache_file}"
                    )
                except Exception as e:
                    logger.warning(f"Could not save cache: {e}")

            # Log progress periodically
            if i % 100 == 0:
                logger.info(
                    f"Processed {i}/{len(bugs)} bugs ({i/len(bugs)*100:.1f}%) - "
                    f"Cache keys: {successful_cache_keys} success, {failed_cache_keys} failed"
                )

        # Save final cache
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(merged_results, f, indent=2, ensure_ascii=False)
            logger.info(
                f"Saved final cache with {len(merged_results)} results to {cache_file}"
            )
        except Exception as e:
            logger.warning(f"Could not save final cache: {e}")

        logger.info(f"Completed processing {len(merged_results)} bugs")
        logger.info(
            f"Cache key generation: {successful_cache_keys} success, {failed_cache_keys} failed"
        )
        return merged_results

    def save_results(self, results: Dict[str, Dict[str, Any]]):
        """Save merged results to JSON file"""
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(
                f"Saved {len(results)} merged bug results to {self.output_file}"
            )
        except IOError as e:
            logger.error(f"Failed to save results: {e}")

    def generate_statistics(self, results: Dict[str, Dict[str, Any]]):
        """Generate and log statistics about the merged results"""
        total_bugs = len(results)
        bug_related_count = sum(1 for data in results.values()
                                if data.get('is_bug_related') is True)
        not_bug_related_count = sum(1 for data in results.values()
                                    if data.get('is_bug_related') is False)
        unknown_count = sum(1 for data in results.values()
                            if data.get('is_bug_related') is None)

        status_counts = {}
        for data in results.values():
            status = data.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1

        logger.info("=== MERGE STATISTICS ===")
        logger.info(f"Total bugs processed: {total_bugs}")
        logger.info(
            f"Bug-related: {bug_related_count} ({bug_related_count/total_bugs*100:.1f}%)"
        )
        logger.info(
            f"Not bug-related: {not_bug_related_count} ({not_bug_related_count/total_bugs*100:.1f}%)"
        )
        logger.info(
            f"Unknown/missing analysis: {unknown_count} ({unknown_count/total_bugs*100:.1f}%)"
        )
        logger.info("Status distribution:")
        for status, count in sorted(status_counts.items()):
            logger.info(f"  {status}: {count} ({count/total_bugs*100:.1f}%)")

    def run(self):
        """Main processing function"""
        logger.info("Starting bug results merge process...")

        # Validate input files
        for file_path in [self.llm_bugs_file, self.analysis_cache_file]:
            if not Path(file_path).exists():
                logger.error(f"Input file not found: {file_path}")
                return

        # Process all bugs
        merged_results = self.process_all_bugs()

        if not merged_results:
            logger.error("No results to save")
            return

        # Save results
        self.save_results(merged_results)

        # Generate statistics
        self.generate_statistics(merged_results)

        logger.info("Bug results merge completed successfully!")


def main():
    """Main function"""
    llm_bugs_file = "llm_bugs.json"
    analysis_cache_file = "analysis_cache.json"
    output_file = "merged_bug_results.json"

    merger = BugResultsMerger(llm_bugs_file, analysis_cache_file, output_file)

    try:
        merger.run()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
