#!/usr/bin/env python3
"""
Script to merge bug analysis results from multiple sources:
- llm_bugs.json: contains title, description, status
- analysis_cache.json: contains is_bug_related, reasoning, confidence
- URL to cache key mapping logic from collect_issues.py
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from urllib.parse import urlparse

from google.generativeai.protos import Candidate
from common import GitHubAPIClient
import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler('data/merge_bug_results.log'),
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

        # Preserve original repo if not in the mapping
        repo = repo_name_map.get(repo, repo)

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
        cache_file = "data/merged_results_cache.json"

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

    def add_timestamps_to_cache(self) -> Dict[str, Dict[str, Any]]:
        """
        Add timestamps to all items in the analysis cache by fetching GitHub data
        Returns: {"url": {"cache_key": "xxx", "timestamp": "xxx"}}
        """
        logger.info("Starting timestamp addition process...")

        # Get all cache items
        cache_items = self.analysis_cache_data.get('items', {})
        total_items = len(cache_items)
        logger.info(f"Processing {total_items} cache items for timestamps...")

        timestamp_results = {}
        successful_timestamps = 0
        failed_timestamps = 0

        # Process each cache item
        for i, (cache_key, cache_data) in enumerate(cache_items.items(), 1):
            try:
                # Extract URL from cache data if available, or try to reconstruct from cache key
                url = cache_data.get('url', '')

                if not url:
                    # Try to reconstruct URL from cache key
                    # Cache key format: "owner/repo:issue|pr:number"
                    if ':' in cache_key:
                        parts = cache_key.split(':')
                        if len(parts) == 3:
                            repo_part, item_type, item_number = parts
                            if item_type == 'issue':
                                url = f"https://github.com/{repo_part}/issues/{item_number}"
                            elif item_type == 'pr':
                                url = f"https://github.com/{repo_part}/pull/{item_number}"

                if not url:
                    logger.warning(
                        f"Could not determine URL for cache key: {cache_key}")
                    failed_timestamps += 1
                    continue

                # Fetch GitHub data to get timestamp
                item_data = self.get_github_item_data(url)
                if item_data:
                    # Extract creation timestamp
                    created_at = item_data.get('created_at', '')
                    if created_at:
                        timestamp_results[url] = {
                            'cache_key': cache_key,
                            'timestamp': created_at
                        }
                        successful_timestamps += 1
                        logger.debug(
                            f"Added timestamp for {url}: {created_at}")
                    else:
                        logger.warning(
                            f"No creation timestamp found for {url}")
                        failed_timestamps += 1
                else:
                    logger.warning(f"Failed to fetch GitHub data for {url}")
                    failed_timestamps += 1

                # Log progress periodically
                if i % 100 == 0:
                    logger.info(
                        f"Processed {i}/{total_items} cache items ({i/total_items*100:.1f}%) - "
                        f"Timestamps: {successful_timestamps} success, {failed_timestamps} failed"
                    )

            except Exception as e:
                logger.error(f"Error processing cache key '{cache_key}': {e}")
                failed_timestamps += 1

        logger.info(f"Completed timestamp addition process")
        logger.info(f"Successful timestamps: {successful_timestamps}")
        logger.info(f"Failed timestamps: {failed_timestamps}")

        return timestamp_results

    def save_timestamp_results(self, timestamp_results: Dict[str, Dict[str,
                                                                       Any]],
                               output_file: str):
        """Save timestamp results to JSON file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(timestamp_results, f, indent=2, ensure_ascii=False)
            logger.info(
                f"Saved {len(timestamp_results)} timestamp results to {output_file}"
            )
        except IOError as e:
            logger.error(f"Failed to save timestamp results: {e}")

    def add_timestamps_by_iterating_github(
            self,
            iter_cache_file: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Iterate over issues and PRs from repos present in the analysis cache, build
        cache keys using the same logic, and if the key exists in the cache, record
        the creation timestamp.

        Returns a mapping: {"<html_url>": {"cache_key": "<key>", "timestamp": "<created_at>"}}
        """
        logger.info(
            "Starting timestamp addition by iterating GitHub issues/PRs...")

        cache_items: Dict[str, Any] = self.analysis_cache_data.get('items', {})
        if not cache_items:
            logger.warning(
                "No items found in analysis cache; nothing to process.")
            return {}

        # Build repo -> set(cache_keys) for quick membership checks and early stopping per repo
        repo_to_keys: Dict[str, set] = {}
        for cache_key in cache_items.keys():
            # cache key format: "owner/repo:issue|pr:id"
            if ':' not in cache_key:
                continue
            repo_part = cache_key.split(':', 1)[0]
            repo_to_keys.setdefault(repo_part, set()).add(cache_key)

        logger.info(
            f"Found {len(cache_items)} cache items across {len(repo_to_keys)} repos"
        )

        # Load previous progress if cache provided
        results: Dict[str, Dict[str, Any]] = {}
        repo_progress: Dict[str, int] = {}
        if iter_cache_file and os.path.exists(iter_cache_file):
            try:
                with open(iter_cache_file, 'r', encoding='utf-8') as f:
                    cache_payload = json.load(f)
                    results = cache_payload.get('results', {})
                    repo_progress = cache_payload.get('repo_progress', {})
                logger.info(
                    f"Loaded iterate cache with {len(results)} results from {iter_cache_file}"
                )
            except Exception as e:
                logger.warning(
                    f"Could not load iterate cache file: {e}. Falling back to fresh run."
                )
                results = {}
                repo_progress = {}

        # Overall progress counters
        overall_total = sum(len(keys) for keys in repo_to_keys.values())
        overall_matched = 0
        logger.info(
            f"Iterate mode progress: {overall_matched}/{overall_total} matched (0.0%)"
        )

        total_matched = len(results)

        for repo_full, remaining_keys in repo_to_keys.items():
            owner_repo = repo_full
            # Skip keys already present in results for this repo
            if results:
                try:
                    matched_keys_for_repo = {
                        v.get('cache_key')
                        for v in results.values()
                        if isinstance(v, dict) and str(v.get(
                            'cache_key', '')).startswith(f"{owner_repo}:")
                    }
                    before_len = len(remaining_keys)
                    remaining_keys.difference_update(
                        k for k in matched_keys_for_repo if k)
                    pruned_count = before_len - len(remaining_keys)
                    if pruned_count > 0:
                        overall_matched += pruned_count
                except Exception:
                    pass
            repo_initial_total = len(remaining_keys)
            logger.info(
                f"Processing repo {owner_repo} with {repo_initial_total} targets; overall {overall_matched}/{overall_total} matched"
            )

            # GraphQL-only pagination for issues and pull requests
            owner, repo = owner_repo.split('/', 1)

            # Load resume state if available
            progress_state = repo_progress.get(owner_repo, {}) if isinstance(
                repo_progress.get(owner_repo, {}), dict) else {}
            issues_cursor = progress_state.get('issues_cursor')
            prs_cursor = progress_state.get('prs_cursor')
            issues_done = progress_state.get('issues_done', False)
            prs_done = progress_state.get('prs_done', False)

            # Iterate issues first
            if not issues_done and remaining_keys:
                while True:
                    gql_issues = self.github_client.graphql_list_issues(
                        owner, repo, page_size=100, after=issues_cursor)
                    if not gql_issues:
                        break
                    issues_nodes = (gql_issues.get('data', {}).get(
                        'repository', {}).get('issues', {}))
                    nodes = issues_nodes.get('nodes', [])
                    pageInfo = issues_nodes.get('pageInfo', {})

                    matched_in_page = 0
                    for node in nodes:
                        item_data = {
                            'id': node.get('databaseId'),
                        }
                        cache_key = self.generate_cache_key(
                            item_data, owner_repo)
                        if cache_key in remaining_keys:
                            html_url = node.get('url', '')
                            created_at = node.get('createdAt', '')
                            if html_url and created_at:
                                results[html_url] = {
                                    'cache_key': cache_key,
                                    'timestamp': created_at
                                }
                                total_matched += 1
                                matched_in_page += 1
                                overall_matched += 1
                                remaining_keys.remove(cache_key)

                    # Update cursor and save progress
                    issues_cursor = pageInfo.get('endCursor')
                    if iter_cache_file:
                        try:
                            tmp_path = f"{iter_cache_file}.tmp"
                            progress_state.update({
                                'issues_cursor': issues_cursor,
                                'prs_cursor': prs_cursor,
                                'issues_done': False,
                                'prs_done': prs_done,
                            })
                            repo_progress[owner_repo] = progress_state
                            payload = {
                                'results': results,
                                'repo_progress': repo_progress
                            }
                            with open(tmp_path, 'w', encoding='utf-8') as f:
                                json.dump(payload,
                                          f,
                                          indent=2,
                                          ensure_ascii=False)
                            os.replace(tmp_path, iter_cache_file)
                            repo_matched = (repo_initial_total -
                                            len(remaining_keys))
                            repo_pct = (repo_matched / repo_initial_total *
                                        100) if repo_initial_total else 100.0
                            overall_pct = (overall_matched / overall_total *
                                           100) if overall_total else 100.0
                            logger.info(
                                f"[GQL issues {owner_repo}] +{matched_in_page} matched, repo {repo_matched}/{repo_initial_total} ({repo_pct:.1f}%), overall {overall_matched}/{overall_total} ({overall_pct:.1f}%). Progress saved."
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to write iterate cache (issues): {e}")

                    if not remaining_keys:
                        logger.info(
                            f"All cached items matched for {owner_repo} (issues)"
                        )
                        break

                    if not pageInfo.get('hasNextPage'):
                        issues_done = True
                        progress_state['issues_done'] = True
                        break

            # Then iterate pull requests
            if not prs_done and remaining_keys:
                while True:
                    gql_prs = self.github_client.graphql_list_pull_requests(
                        owner, repo, page_size=100, after=prs_cursor)
                    if not gql_prs:
                        break
                    prs_nodes = (gql_prs.get('data',
                                             {}).get('repository', {}).get(
                                                 'pullRequests', {}))
                    nodes = prs_nodes.get('nodes', [])
                    pageInfo = prs_nodes.get('pageInfo', {})

                    matched_in_page = 0
                    for node in nodes:
                        # Candidate 1: use PR databaseId (GraphQL pullRequest.databaseId)
                        pr_item_data = {
                            'id': node.get('databaseId'),
                            'pull_request': True,
                        }
                        cache_key_candidate = self.generate_cache_key(
                            pr_item_data, owner_repo)

                        selected_cache_key = cache_key_candidate
                        if selected_cache_key not in remaining_keys:
                            pr_item_data = {
                                'id': node.get('databaseId'),
                            }
                            selected_cache_key = self.generate_cache_key(
                                pr_item_data, owner_repo)

                        if selected_cache_key in remaining_keys:
                            html_url = node.get('url', '')
                            created_at = node.get('createdAt', '')
                            if html_url and created_at:
                                results[html_url] = {
                                    'cache_key': selected_cache_key,
                                    'timestamp': created_at
                                }
                                total_matched += 1
                                matched_in_page += 1
                                overall_matched += 1
                                remaining_keys.remove(selected_cache_key)

                    # Update cursor and save progress
                    prs_cursor = pageInfo.get('endCursor')
                    if iter_cache_file:
                        try:
                            tmp_path = f"{iter_cache_file}.tmp"
                            progress_state.update({
                                'issues_cursor': issues_cursor,
                                'prs_cursor': prs_cursor,
                                'issues_done': issues_done,
                                'prs_done': False,
                            })
                            repo_progress[owner_repo] = progress_state
                            payload = {
                                'results': results,
                                'repo_progress': repo_progress
                            }
                            with open(tmp_path, 'w', encoding='utf-8') as f:
                                json.dump(payload,
                                          f,
                                          indent=2,
                                          ensure_ascii=False)
                            os.replace(tmp_path, iter_cache_file)
                            repo_matched = (repo_initial_total -
                                            len(remaining_keys))
                            repo_pct = (repo_matched / repo_initial_total *
                                        100) if repo_initial_total else 100.0
                            overall_pct = (overall_matched / overall_total *
                                           100) if overall_total else 100.0
                            logger.info(
                                f"[GQL PRs {owner_repo}] +{matched_in_page} matched, repo {repo_matched}/{repo_initial_total} ({repo_pct:.1f}%), overall {overall_matched}/{overall_total} ({overall_pct:.1f}%). Progress saved."
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to write iterate cache (pull requests): {e}"
                            )

                    if not remaining_keys:
                        logger.info(
                            f"All cached items matched for {owner_repo} (pull requests)"
                        )
                        break

                    if not pageInfo.get('hasNextPage'):
                        prs_done = True
                        progress_state['prs_done'] = True
                        break

            if remaining_keys:
                logger.info(
                    f"Unmatched items for {owner_repo}: {len(remaining_keys)} remain after pagination"
                )
                logger.info(list(remaining_keys)[:20])

        logger.info(
            f"Timestamp addition complete via GitHub iteration. Matched {total_matched} items."
        )

        # Final save of cache file
        if iter_cache_file:
            try:
                tmp_path = f"{iter_cache_file}.tmp"
                payload = {'results': results, 'repo_progress': repo_progress}
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
                os.replace(tmp_path, iter_cache_file)
                logger.info(f"Saved iterate cache to {iter_cache_file}")
            except Exception as e:
                logger.warning(f"Failed to write final iterate cache: {e}")

        return results

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
    parser = argparse.ArgumentParser(
        description='Merge bug analysis results and optionally add timestamps')
    parser.add_argument(
        '--add-timestamps',
        action='store_true',
        help='Add timestamps by looking up each cache item URL')
    parser.add_argument(
        '--add-timestamps-iterate',
        action='store_true',
        help=
        'Iterate GitHub issues/PRs per repo and map to cache keys (uses built-in resume cache)'
    )
    parser.add_argument(
        '--llm-bugs',
        default="data/llm_bugs.json",
        help='Path to llm_bugs.json file (default: data/llm_bugs.json)')
    # Analysis cache path is hardcoded for simplicity per user request
    parser.add_argument(
        '--output',
        default="data/merged_bug_results.json",
        help='Output file path (default: data/merged_bug_results.json)')
    parser.add_argument(
        '--timestamp-output',
        default="data/timestamp_results.json",
        help=
        'Timestamp results output file path (default: data/timestamp_results.json)'
    )

    args = parser.parse_args()

    llm_bugs_file = args.llm_bugs
    analysis_cache_file = "data/analysis_cache.json"
    output_file = args.output
    timestamp_output_file = args.timestamp_output
    iterate_cache_file = "data/timestamp_iter_cache.json"

    merger = BugResultsMerger(llm_bugs_file, analysis_cache_file, output_file)

    try:
        if args.add_timestamps_iterate:
            # Iterate GitHub issues/PRs to build timestamp mapping
            logger.info("Running in timestamp-iterate mode...")
            timestamp_results = merger.add_timestamps_by_iterating_github(
                iter_cache_file=iterate_cache_file)
            if timestamp_results:
                # Save to hardcoded output per user request
                merger.save_timestamp_results(
                    timestamp_results,
                    "data/analysis_cache_with_timestamp.json")
                logger.info(
                    "Timestamp addition (iterate) completed successfully!")
            else:
                logger.warning("No timestamp results to save (iterate mode)")
        elif args.add_timestamps:
            # Only add timestamps
            logger.info("Running in timestamp-only mode...")
            timestamp_results = merger.add_timestamps_to_cache()
            if timestamp_results:
                # Keep existing output flag behavior for this mode
                merger.save_timestamp_results(timestamp_results,
                                              timestamp_output_file)
                logger.info("Timestamp addition completed successfully!")
            else:
                logger.warning("No timestamp results to save")
        else:
            # Run normal merge process
            merger.run()

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
