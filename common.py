#!/usr/bin/env python3
"""
Common utility functions for LLM bug analysis scripts
"""

import requests
import time
import logging
import os
from datetime import datetime, timezone
from urllib.parse import urlparse
from typing import Optional, Dict, Any


logger = logging.getLogger(__name__)


class GitHubAPIClient:
    """
    GitHub API client with rate limiting and authentication
    
    Usage:
        # Basic usage (uses GITHUB_TOKEN env var)
        client = GitHubAPIClient()
        content = client.fetch_url_content("https://github.com/owner/repo/issues/123")
        
        # Batch processing (efficient rate limiting)
        for url in issue_urls:
            content = client.fetch_url_content(url)
    """
    
    def __init__(self, github_token: Optional[str] = None):
        self.session = requests.Session()
        self.github_token = github_token or os.getenv('GITHUB_TOKEN')
        
        if self.github_token:
            self.session.headers.update({
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'LLM-Bug-Analysis-Tool/1.0'
            })
            logger.info("GitHub token found - using authenticated requests")
        else:
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Linux; x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
            logger.warning("No GitHub token found - using unauthenticated requests (much slower)")
        
        # Rate limiting settings - optimized delays
        self.request_delay = 0.05 if self.github_token else 0.2  # Minimal delay for non-API requests
        self.github_api_delay = 0.05 if self.github_token else 0.5  # Minimal delay for API requests
        
        # Rate limit tracking
        self.api_calls_made = 0
        self.rate_limit_remaining = 5000 if self.github_token else 60  # Authenticated vs unauthenticated
        self.rate_limit_reset_time = None
    
    def check_rate_limit(self, response: requests.Response):
        """Update rate limit tracking from response headers"""
        if 'X-RateLimit-Remaining' in response.headers:
            self.rate_limit_remaining = int(response.headers['X-RateLimit-Remaining'])
            self.rate_limit_reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
            
            # Log rate limit status periodically
            if self.api_calls_made % 100 == 0:
                reset_time = datetime.fromtimestamp(self.rate_limit_reset_time, tz=timezone.utc)
                logger.info(f"Rate limit: {self.rate_limit_remaining} remaining, resets at {reset_time}")
    
    def wait_for_rate_limit(self):
        """Wait if we're approaching rate limits - optimized thresholds"""
        # Reduced buffer since we're tracking more accurately
        min_remaining = 10 if self.github_token else 3
        if self.rate_limit_remaining < min_remaining and self.rate_limit_reset_time:
            current_time = datetime.now(timezone.utc).timestamp()
            if current_time < self.rate_limit_reset_time:
                wait_time = self.rate_limit_reset_time - current_time + 5  # Reduced buffer
                logger.warning(f"Rate limit almost exhausted. Waiting {wait_time:.0f} seconds...")
                time.sleep(wait_time)
                self.rate_limit_remaining = 5000 if self.github_token else 60  # Reset counter

    def _is_github_issue_url(self, parsed_url: urlparse) -> bool:
        """Check if URL is a GitHub issue URL"""
        path_parts = parsed_url.path.strip('/').split('/')
        return (len(path_parts) >= 4 and 
                path_parts[2] == 'issues')
    
    def _is_github_pull_request_url(self, parsed_url: urlparse) -> bool:
        """Check if URL is a GitHub pull request URL"""
        path_parts = parsed_url.path.strip('/').split('/')
        return (len(path_parts) >= 4 and 
                path_parts[2] == 'pull')
    
    def _extract_github_path_parts(self, parsed_url: urlparse) -> Optional[tuple]:
        """Extract owner, repo, type, and number from GitHub URL"""
        path_parts = parsed_url.path.strip('/').split('/')
        if len(path_parts) >= 4:
            return path_parts[0], path_parts[1], path_parts[2], path_parts[3]
        return None
    
    def _fetch_github_issue_data(self, owner: str, repo: str, issue_number: str) -> Optional[Dict[str, Any]]:
        """Fetch GitHub issue data via API with unlimited retries for rate limiting"""
        api_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
        attempt = 0
        
        while True:
            attempt += 1
            try:
                # Single optimized rate limit check
                self.wait_for_rate_limit()
                if self.github_token:
                    time.sleep(self.github_api_delay)  # Only minimal delay for authenticated requests
                
                response = self.session.get(api_url, timeout=30)
                self.api_calls_made += 1
                self.check_rate_limit(response)
                
                if response.status_code == 200:
                    return response.json()
                
                elif response.status_code in [403, 429]:
                    # Rate limited - wait and retry indefinitely
                    wait_time = min(attempt * 30, 300)  # 30s, 60s, 90s... up to 5 min max
                    logger.warning(f"Rate limited ({response.status_code}) for issue data, waiting {wait_time}s before retry {attempt}")
                    time.sleep(wait_time)
                    continue
                
                elif response.status_code == 404:
                    # Issue not found
                    logger.warning(f"Issue not found (404) for {api_url}")
                    return None
                
                else:
                    # Other errors - limited retries
                    if attempt >= 5:
                        logger.error(f"GitHub API error {response.status_code} after {attempt} attempts, giving up")
                        return None
                    
                    wait_time = attempt * 30  # 30s, 60s, 90s, 120s, 150s
                    logger.warning(f"GitHub API error {response.status_code}, waiting {wait_time}s before retry {attempt}/5")
                    time.sleep(wait_time)
                    continue
                    
            except requests.exceptions.RequestException as e:
                if attempt >= 5:
                    logger.error(f"Request exception for issue data after {attempt} attempts: {e}")
                    return None
                
                wait_time = attempt * 30
                logger.warning(f"Request exception for issue data: {e}, waiting {wait_time}s before retry {attempt}/5")
                time.sleep(wait_time)
                continue
            except Exception as e:
                logger.debug(f"Unexpected error fetching issue data: {e}")
                return None
    
    def _format_issue_basic_info(self, data: Dict[str, Any]) -> str:
        """Format basic issue information"""
        content = f"Title: {data.get('title', '')}\n\n"
        content += f"Body: {data.get('body', '')}\n\n"
        content += f"State: {data.get('state', '')}\n"
        content += f"Created: {data.get('created_at', '')}\n"
        content += f"Updated: {data.get('updated_at', '')}\n"
        content += f"Labels: {', '.join([label['name'] for label in data.get('labels', [])])}\n"
        return content
    
    def _format_timeline_event(self, event: Dict[str, Any]) -> str:
        """Format a single timeline event"""
        event_type = event.get('event', 'unknown')
        created_at = event.get('created_at', '')
        actor = event.get('actor', {}).get('login', 'unknown') if event.get('actor') else 'system'
        
        if event_type == 'commented':
            body = event.get('body', '')[:300]
            return f"  [{created_at}] {actor} commented: {body}...\n"
        elif event_type == 'closed':
            commit_id = event.get('commit_id', '')
            if commit_id:
                return f"  [{created_at}] {actor} closed this as completed in {commit_id[:7]}\n"
            else:
                return f"  [{created_at}] {actor} closed this as completed\n"
        elif event_type == 'reopened':
            return f"  [{created_at}] {actor} reopened this\n"
        elif event_type == 'referenced':
            commit_id = event.get('commit_id', '')
            if commit_id:
                return f"  [{created_at}] {actor} referenced this in commit {commit_id[:7]}\n"
            else:
                return f"  [{created_at}] {actor} referenced this\n"
        elif event_type == 'cross-referenced':
            source = event.get('source', {})
            if source and source.get('issue'):
                source_url = source['issue'].get('html_url', '')
                source_title = source['issue'].get('title', '')[:50]
                return f"  [{created_at}] Referenced in: {source_title}... ({source_url})\n"
            else:
                return f"  [{created_at}] Cross-referenced\n"
        elif event_type == 'labeled':
            label_name = event.get('label', {}).get('name', 'unknown')
            return f"  [{created_at}] {actor} added label: {label_name}\n"
        elif event_type == 'unlabeled':
            label_name = event.get('label', {}).get('name', 'unknown')
            return f"  [{created_at}] {actor} removed label: {label_name}\n"
        elif event_type == 'assigned':
            assignee = event.get('assignee', {}).get('login', 'unknown')
            return f"  [{created_at}] {actor} assigned {assignee}\n"
        elif event_type == 'mentioned':
            return f"  [{created_at}] {actor} mentioned this\n"
        elif event_type == 'committed':
            commit_sha = event.get('sha', '')[:7]
            commit_message = event.get('message', '')[:100]
            return f"  [{created_at}] Commit {commit_sha}: {commit_message}...\n"
        else:
            return f"  [{created_at}] {actor} {event_type}\n"
    
    def _fetch_timeline_events(self, owner: str, repo: str, issue_number: str) -> Optional[str]:
        """Fetch and format timeline events for an issue with retry logic for rate limiting"""
        timeline_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/timeline"
        attempt = 0
        
        while True:
            attempt += 1
            try:
                # Single optimized rate limit check
                self.wait_for_rate_limit()
                if self.github_token:
                    time.sleep(self.github_api_delay)  # Only minimal delay for authenticated requests
                
                timeline_response = self.session.get(
                    timeline_url, 
                )
                self.api_calls_made += 1
                self.check_rate_limit(timeline_response)
                
                if timeline_response.status_code == 200:
                    timeline_data = timeline_response.json()
                    if timeline_data:
                        content = f"\nTimeline Events ({len(timeline_data)}):\n"
                        
                        for i, event in enumerate(timeline_data):
                            content += self._format_timeline_event(event)
                            
                            # Limit to prevent extremely long content
                            if i >= 50:
                                content += f"  ... (showing first 50 of {len(timeline_data)} timeline events)\n"
                                break
                        
                        return content
                    else:
                        logger.debug(f"No timeline data found for {timeline_url}")
                        return None
                
                elif timeline_response.status_code in [403, 429]:
                    # Rate limited - wait and retry indefinitely
                    wait_time = min(attempt * 30, 300)  # 30s, 60s, 90s... up to 5 min max
                    logger.warning(f"Rate limited ({timeline_response.status_code}) for timeline, waiting {wait_time}s before retry {attempt}")
                    time.sleep(wait_time)
                    continue
                
                elif timeline_response.status_code == 404:
                    # Timeline not available for this issue
                    logger.debug(f"Timeline not available (404) for {timeline_url}")
                    return None
                
                else:
                    # Other errors - limited retries
                    if attempt >= 5:
                        logger.error(f"Timeline API error {timeline_response.status_code} after {attempt} attempts, giving up")
                        return None
                    
                    wait_time = attempt * 30  # 30s, 60s, 90s, 120s, 150s
                    logger.warning(f"Timeline API error {timeline_response.status_code}, waiting {wait_time}s before retry {attempt}/5")
                    time.sleep(wait_time)
                    continue
                    
            except requests.exceptions.RequestException as e:
                if attempt >= 5:
                    logger.error(f"Request exception for timeline after {attempt} attempts: {e}")
                    return None
                
                wait_time = attempt * 30
                logger.warning(f"Request exception for timeline: {e}, waiting {wait_time}s before retry {attempt}/5")
                time.sleep(wait_time)
                continue
            except Exception as e:
                logger.debug(f"Unexpected error fetching timeline: {e}")
                return None

    def api_get(self, url: str, headers: Optional[Dict[str, str]] = None) -> Optional[requests.Response]:
        """
        Make a GitHub API GET request with unlimited retries for rate limiting
        
        Args:
            url: The GitHub API URL to fetch
            headers: Optional headers to include in the request
            
        Returns:
            Response object if successful, None if failed after retries
        """
        attempt = 0
        while True:
            attempt += 1
            try:
                # Single optimized rate limit check
                self.wait_for_rate_limit()
                if self.github_token:
                    time.sleep(self.github_api_delay)  # Only minimal delay for authenticated requests
                
                response = self.session.get(url, headers=headers, timeout=30)
                self.api_calls_made += 1
                self.check_rate_limit(response)
                
                if response.status_code == 200:
                    return response
                
                elif response.status_code in [403, 429]:
                    # Rate limited - wait and retry indefinitely
                    wait_time = min(attempt * 30, 300)  # 30s, 60s, 90s... up to 5 min max
                    logger.warning(f"Rate limited ({response.status_code}) for API request, waiting {wait_time}s before retry {attempt}")
                    time.sleep(wait_time)
                    continue
                
                elif response.status_code == 404:
                    # Resource not found
                    logger.warning(f"Resource not found (404) for {url}")
                    return None
                
                else:
                    # Other errors - limited retries
                    if attempt >= 5:
                        logger.error(f"GitHub API error {response.status_code} after {attempt} attempts, giving up")
                        return None
                    
                    wait_time = attempt * 30  # 30s, 60s, 90s, 120s, 150s
                    logger.warning(f"GitHub API error {response.status_code}, waiting {wait_time}s before retry {attempt}/5")
                    time.sleep(wait_time)
                    continue
                    
            except requests.exceptions.RequestException as e:
                if attempt >= 5:
                    logger.error(f"Request exception after {attempt} attempts: {e}")
                    return None
                
                wait_time = attempt * 30
                logger.warning(f"Request exception: {e}, waiting {wait_time}s before retry {attempt}/5")
                time.sleep(wait_time)
                continue
            except Exception as e:
                logger.debug(f"Unexpected error in API request: {e}")
                return None

    
    def _fetch_github_issue_content(self, url: str) -> str:
        """Fetch content for a GitHub issue URL"""
        parsed_url = urlparse(url)
        path_parts = self._extract_github_path_parts(parsed_url)
        if not path_parts:
            return f"ERROR: Invalid GitHub URL format - {url}"
        
        owner, repo, item_type, issue_number = path_parts
        
        # Fetch issue data
        data = self._fetch_github_issue_data(owner, repo, issue_number)
        if not data:
            return f"ERROR: Failed to fetch GitHub issue data for {url}"
        
        # Format basic info
        content = self._format_issue_basic_info(data)
        
        # Try to get timeline events (with retry logic for rate limiting)
        timeline_content = self._fetch_timeline_events(owner, repo, issue_number)
        if timeline_content:
            content += timeline_content
        
        return content
    
    def _fetch_regular_url_content(self, url: str) -> str:
        """Fetch content for non-GitHub URLs"""
        time.sleep(self.request_delay)
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        return response.text
    
    def _fetch_github_pull_request_data(self, owner: str, repo: str, pr_number: str) -> Optional[Dict[str, Any]]:
        """Fetch GitHub pull request data via API with unlimited retries for rate limiting"""
        api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
        attempt = 0
        
        while True:
            attempt += 1
            try:
                # Single optimized rate limit check
                self.wait_for_rate_limit()
                if self.github_token:
                    time.sleep(self.github_api_delay)  # Only minimal delay for authenticated requests
                
                response = self.session.get(api_url, timeout=30)
                self.api_calls_made += 1
                self.check_rate_limit(response)
                
                if response.status_code == 200:
                    return response.json()
                
                elif response.status_code in [403, 429]:
                    # Rate limited - wait and retry indefinitely
                    wait_time = min(attempt * 30, 300)  # 30s, 60s, 90s... up to 5 min max
                    logger.warning(f"Rate limited ({response.status_code}) for PR data, waiting {wait_time}s before retry {attempt}")
                    time.sleep(wait_time)
                    continue
                
                elif response.status_code == 404:
                    # Pull request not found
                    logger.warning(f"Pull request not found (404) for {api_url}")
                    return None
                
                else:
                    # Other errors - limited retries
                    if attempt >= 5:
                        logger.error(f"GitHub API error {response.status_code} after {attempt} attempts, giving up")
                        return None
                    
                    wait_time = attempt * 30  # 30s, 60s, 90s, 120s, 150s
                    logger.warning(f"GitHub API error {response.status_code}, waiting {wait_time}s before retry {attempt}/5")
                    time.sleep(wait_time)
                    continue
                    
            except requests.exceptions.RequestException as e:
                if attempt >= 5:
                    logger.error(f"Request exception for PR data after {attempt} attempts: {e}")
                    return None
                
                wait_time = attempt * 30
                logger.warning(f"Request exception for PR data: {e}, waiting {wait_time}s before retry {attempt}/5")
                time.sleep(wait_time)
                continue
            except Exception as e:
                logger.debug(f"Unexpected error fetching PR data: {e}")
                return None
    
    def get_pull_request_status(self, url: str) -> str:
        """
        Get the status of a GitHub pull request
        
        Args:
            url: The GitHub pull request URL (e.g., https://github.com/owner/repo/pull/123)
            
        Returns:
            String status: 'Open', 'Draft', 'Closed', 'Merged', or 'Other'
            
        Raises:
            ValueError: If the URL is not a valid GitHub pull request URL
            RuntimeError: If there's an error fetching the pull request data
        """
        parsed_url = urlparse(url)
        
        # Validate it's a GitHub pull request URL
        if parsed_url.hostname != 'github.com':
            error_msg = f"Not a GitHub URL: {url}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if not self._is_github_pull_request_url(parsed_url):
            error_msg = f"Not a GitHub pull request URL: {url}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Extract path parts
        path_parts = self._extract_github_path_parts(parsed_url)
        if not path_parts:
            error_msg = f"Invalid GitHub URL format: {url}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        owner, repo, item_type, pr_number = path_parts
        
        # Fetch pull request data
        data = self._fetch_github_pull_request_data(owner, repo, pr_number)
        if not data:
            error_msg = f"Failed to fetch pull request data for {url}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Determine status based on GitHub's PR status values
        state = data.get('state', 'unknown')
        merged = data.get('merged', False)
        draft = data.get('draft', False)
        
        if merged:
            return 'Merged'
        elif draft:
            return 'Draft'
        elif state == 'closed':
            return 'Closed'
        elif state == 'open':
            return 'Open'
        else:
            # Only return 'Other' for unknown status values, not for errors
            logger.warning(f"Unknown PR status: state={state}, merged={merged}, draft={draft} for {url}")
            return 'Other'
    
    def get_pull_request_diff(self, url: str) -> str:
        """
        Get the diff content of a GitHub pull request
        
        Args:
            url: The GitHub pull request URL (e.g., https://github.com/owner/repo/pull/123)
            
        Returns:
            String containing the diff content of the pull request
            
        Raises:
            ValueError: If the URL is not a valid GitHub pull request URL
            RuntimeError: If there's an error fetching the pull request diff
        """
        parsed_url = urlparse(url)
        
        # Validate it's a GitHub pull request URL
        if parsed_url.hostname != 'github.com':
            error_msg = f"Not a GitHub URL: {url}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if not self._is_github_pull_request_url(parsed_url):
            error_msg = f"Not a GitHub pull request URL: {url}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Extract path parts
        path_parts = self._extract_github_path_parts(parsed_url)
        if not path_parts:
            error_msg = f"Invalid GitHub URL format: {url}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        owner, repo, item_type, pr_number = path_parts
        
        # Fetch pull request diff using GitHub API
        diff_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
        
        # Use the existing api_get method with Accept header for diff format
        headers = {'Accept': 'application/vnd.github.diff'}
        response = self.api_get(diff_url, headers=headers)
        
        if not response:
            error_msg = f"Failed to fetch pull request diff for {url}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        return response.text
    
    def fetch_url_content(self, url: str) -> str:
        """
        Fetch content from a URL with error handling and rate limit management
        
        Args:
            url: The URL to fetch content from
            
        Returns:
            String containing the fetched content or error message
        """
        try:
            parsed_url = urlparse(url)
            
            # Handle GitHub URLs
            if parsed_url.hostname == 'github.com':
                if self._is_github_issue_url(parsed_url):
                    return self._fetch_github_issue_content(url)
                elif self._is_github_pull_request_url(parsed_url):
                    logger.info(f"Skipping pull request: {url}")
                    return "SKIPPED: Pull request - only processing issues"
                else:
                    return self._fetch_regular_url_content(url)
            
            # Handle non-GitHub URLs
            return self._fetch_regular_url_content(url)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return f"ERROR: Failed to fetch content - {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {e}")
            return f"ERROR: Unexpected error - {str(e)}"