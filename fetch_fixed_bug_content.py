#!/usr/bin/env python3
"""
Script to fetch content from URLs of fixed bugs in llm_bugs.json
and save them to fixed_bug_content.json
"""

import json
import requests
import time
import logging
from pathlib import Path
from typing import Dict, List, Any
import sys
from urllib.parse import urlparse
import os
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fetch_bug_content.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class BugContentFetcher:
    def __init__(self, input_file: str, output_file: str):
        self.input_file = input_file
        self.output_file = output_file
        self.session = requests.Session()
        
        # GitHub token setup
        self.github_token = os.getenv('GITHUB_TOKEN')
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
        
        # Rate limiting settings
        self.request_delay = 0.1 if self.github_token else 1.0  # Much faster with token
        self.github_api_delay = 0.5 if self.github_token else 2.0  # Faster API calls with token
        
        # Rate limit tracking
        self.api_calls_made = 0
        self.rate_limit_remaining = 5000 if self.github_token else 60  # Authenticated vs unauthenticated
        self.rate_limit_reset_time = None
        
        # Load existing results if file exists
        self.existing_content = self.load_existing_results()
        
    def load_existing_results(self) -> Dict[str, str]:
        """Load existing results to resume from where we left off"""
        if Path(self.output_file).exists():
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load existing results: {e}")
        return {}
    
    def save_results(self, content_dict: Dict[str, str]):
        """Save results to JSON file"""
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(content_dict, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(content_dict)} entries to {self.output_file}")
        except IOError as e:
            logger.error(f"Failed to save results: {e}")
            
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
        """Wait if we're approaching rate limits"""
        # Need more buffer since we make 2 API calls per issue (issue + timeline)
        min_remaining = 20 if self.github_token else 5
        if self.rate_limit_remaining < min_remaining and self.rate_limit_reset_time:
            current_time = datetime.now(timezone.utc).timestamp()
            if current_time < self.rate_limit_reset_time:
                wait_time = self.rate_limit_reset_time - current_time + 10  # Add 10s buffer
                logger.warning(f"Rate limit almost exhausted. Waiting {wait_time:.0f} seconds...")
                time.sleep(wait_time)
                self.rate_limit_remaining = 5000 if self.github_token else 60  # Reset counter

    def fetch_url_content(self, url: str) -> str:
        """Fetch content from a URL with error handling and rate limit management"""
        try:
            # Check if URL is a GitHub URL for API optimization
            parsed_url = urlparse(url)
            if parsed_url.hostname == 'github.com':
                # For GitHub URLs, we can use the API for better rate limiting
                # Convert github.com/owner/repo/issues/123 to API format (only issues, skip pulls)
                path_parts = parsed_url.path.strip('/').split('/')
                if len(path_parts) >= 4 and path_parts[2] == 'issues':  # Only process issues

                    # Check rate limit before making request
                    self.wait_for_rate_limit()
                    
                    api_url = f"https://api.github.com/repos/{path_parts[0]}/{path_parts[1]}/issues/{path_parts[3]}"
                    time.sleep(self.github_api_delay)
                    
                    response = self.session.get(api_url, timeout=30)
                    self.api_calls_made += 1
                    self.check_rate_limit(response)
                    
                    if response.status_code == 200:
                        data = response.json()
                        # Combine title, body, and comments for comprehensive content
                        content = f"Title: {data.get('title', '')}\n\n"
                        content += f"Body: {data.get('body', '')}\n\n"
                        content += f"State: {data.get('state', '')}\n"
                        content += f"Created: {data.get('created_at', '')}\n"
                        content += f"Updated: {data.get('updated_at', '')}\n"
                        content += f"Labels: {', '.join([label['name'] for label in data.get('labels', [])])}\n"
                        
                        # Try to get timeline events (includes comments, close events, references, etc.)
                        timeline_url = f"https://api.github.com/repos/{path_parts[0]}/{path_parts[1]}/issues/{path_parts[3]}/timeline"
                        if self.rate_limit_remaining > 5:
                            try:
                                self.wait_for_rate_limit()
                                time.sleep(self.github_api_delay)
                                
                                # Use timeline API to get comprehensive event history
                                timeline_response = self.session.get(
                                    timeline_url, 
                                    headers={'Accept': 'application/vnd.github.mockingbird-preview+json'},
                                    timeout=30
                                )
                                self.api_calls_made += 1
                                self.check_rate_limit(timeline_response)
                                
                                if timeline_response.status_code == 200:
                                    timeline_data = timeline_response.json()
                                    if timeline_data:
                                        content += f"\nTimeline Events ({len(timeline_data)}):\n"
                                        
                                        for i, event in enumerate(timeline_data):
                                            event_type = event.get('event', 'unknown')
                                            created_at = event.get('created_at', '')
                                            actor = event.get('actor', {}).get('login', 'unknown') if event.get('actor') else 'system'
                                            
                                            if event_type == 'commented':
                                                # Regular comment
                                                body = event.get('body', '')[:300]  # Truncate long comments
                                                content += f"  [{created_at}] {actor} commented: {body}...\n"
                                                
                                            elif event_type == 'closed':
                                                # Issue closed
                                                commit_id = event.get('commit_id', '')
                                                if commit_id:
                                                    content += f"  [{created_at}] {actor} closed this as completed in {commit_id[:7]}\n"
                                                else:
                                                    content += f"  [{created_at}] {actor} closed this as completed\n"
                                                    
                                            elif event_type == 'reopened':
                                                content += f"  [{created_at}] {actor} reopened this\n"
                                                
                                            elif event_type == 'referenced':
                                                # Referenced in another issue/PR
                                                commit_id = event.get('commit_id', '')
                                                if commit_id:
                                                    content += f"  [{created_at}] {actor} referenced this in commit {commit_id[:7]}\n"
                                                else:
                                                    content += f"  [{created_at}] {actor} referenced this\n"
                                                    
                                            elif event_type == 'cross-referenced':
                                                # Cross-referenced from another issue/PR
                                                source = event.get('source', {})
                                                if source and source.get('issue'):
                                                    source_url = source['issue'].get('html_url', '')
                                                    source_title = source['issue'].get('title', '')[:50]
                                                    content += f"  [{created_at}] Referenced in: {source_title}... ({source_url})\n"
                                                else:
                                                    content += f"  [{created_at}] Cross-referenced\n"
                                                    
                                            elif event_type == 'labeled':
                                                label_name = event.get('label', {}).get('name', 'unknown')
                                                content += f"  [{created_at}] {actor} added label: {label_name}\n"
                                                
                                            elif event_type == 'unlabeled':
                                                label_name = event.get('label', {}).get('name', 'unknown')
                                                content += f"  [{created_at}] {actor} removed label: {label_name}\n"
                                                
                                            elif event_type == 'assigned':
                                                assignee = event.get('assignee', {}).get('login', 'unknown')
                                                content += f"  [{created_at}] {actor} assigned {assignee}\n"
                                                
                                            elif event_type == 'mentioned':
                                                content += f"  [{created_at}] {actor} mentioned this\n"
                                                
                                            elif event_type == 'committed':
                                                commit_sha = event.get('sha', '')[:7]
                                                commit_message = event.get('message', '')[:100]
                                                content += f"  [{created_at}] Commit {commit_sha}: {commit_message}...\n"
                                                
                                            else:
                                                # Other event types
                                                content += f"  [{created_at}] {actor} {event_type}\n"
                                                
                                            # Limit to prevent extremely long content
                                            if i >= 50:  # Limit to first 50 timeline events
                                                content += f"  ... (showing first 50 of {len(timeline_data)} timeline events)\n"
                                                break
                                                
                                else:
                                    logger.debug(f"Timeline API returned {timeline_response.status_code} for {url}")
                                    # Fallback to regular comments API
                                    comments_url = data.get('comments_url')
                                    if comments_url:
                                        self.wait_for_rate_limit()
                                        time.sleep(self.github_api_delay)
                                        comments_response = self.session.get(comments_url, timeout=30)
                                        self.api_calls_made += 1
                                        self.check_rate_limit(comments_response)
                                        
                                        if comments_response.status_code == 200:
                                            comments_data = comments_response.json()
                                            if comments_data:
                                                content += f"\nComments ({len(comments_data)}):\n"
                                                for i, comment in enumerate(comments_data[:5]):
                                                    content += f"Comment {i+1} ({comment.get('created_at', '')}): {comment.get('body', '')[:500]}...\n"
                                                    
                            except Exception as e:
                                logger.debug(f"Failed to fetch timeline for {url}: {e}")
                        
                        return content
                    elif response.status_code == 403:
                        logger.warning(f"Rate limited or forbidden for {url}")
                        return f"ERROR: GitHub API rate limited or forbidden - {response.status_code}"
                    else:
                        logger.warning(f"GitHub API returned {response.status_code} for {url}")
                elif len(path_parts) >= 4 and path_parts[2] == 'pull':
                    # Skip pull requests
                    logger.info(f"Skipping pull request: {url}")
                    return "SKIPPED: Pull request - only processing issues"
            
            # Fallback to regular HTTP request for non-GitHub URLs
            time.sleep(self.request_delay)
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Return the raw HTML content
            return response.text
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return f"ERROR: Failed to fetch content - {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {e}")
            return f"ERROR: Unexpected error - {str(e)}"
    
    def load_bugs_data(self) -> List[Dict[str, Any]]:
        """Load and filter fixed bugs from the JSON file (issues only)"""
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Filter for fixed bugs that are issues (not pull requests)
            fixed_issues = []
            for bug in data['bugs']:
                if bug.get('status') == 'fixed':
                    url = bug.get('url', '')
                    # Only include GitHub issues, skip pull requests
                    if '/issues/' in url or not url.startswith('https://github.com'):
                        fixed_issues.append(bug)
            
            logger.info(f"Found {len(fixed_issues)} fixed issues (excluding pull requests) out of {len(data['bugs'])} total bugs")
            return fixed_issues
            
        except (json.JSONDecodeError, IOError, KeyError) as e:
            logger.error(f"Failed to load bugs data: {e}")
            return []
    
    def process_bugs(self):
        """Main processing function"""
        logger.info("Starting bug content fetching process...")
        
        # Load fixed bugs
        fixed_bugs = self.load_bugs_data()
        if not fixed_bugs:
            logger.error("No fixed bugs found or failed to load data")
            return
        
        # Start with existing content
        content_dict = self.existing_content.copy()
        total_bugs = len(fixed_bugs)
        already_processed = len(content_dict)
        
        logger.info(f"Processing {total_bugs} fixed bugs")
        logger.info(f"Already processed: {already_processed}")
        logger.info(f"Remaining: {total_bugs - already_processed}")
        
        # Process each bug
        for i, bug in enumerate(fixed_bugs, 1):
            url = bug.get('url')
            if not url:
                logger.warning(f"Bug #{i} has no URL, skipping")
                continue
                
            # Skip if already processed
            if url in content_dict:
                if i % 100 == 0:  # Progress update every 100 items
                    logger.info(f"Progress: {i}/{total_bugs} ({i/total_bugs*100:.1f}%) - Skipped (already processed): {url}")
                continue
            
            logger.info(f"Processing {i}/{total_bugs} ({i/total_bugs*100:.1f}%): {url}")
            
            # Fetch content
            content = self.fetch_url_content(url)
            content_dict[url] = content
            
            # Save progress periodically (every 50 requests)
            if i % 50 == 0:
                self.save_results(content_dict)
                logger.info(f"Intermediate save completed. Processed: {len(content_dict)}")
        
        # Final save
        self.save_results(content_dict)
        logger.info(f"Processing complete! Total entries: {len(content_dict)}")


def main():
    """Main function"""
    input_file = "llm_bugs.json"
    output_file = "fixed_bug_content.json"
    
    # Check if input file exists
    if not Path(input_file).exists():
        logger.error(f"Input file {input_file} not found!")
        sys.exit(1)
    
    # Create fetcher and process
    fetcher = BugContentFetcher(input_file, output_file)
    
    try:
        fetcher.process_bugs()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user. Saving current progress...")
        # The periodic saves ensure we don't lose much progress
        logger.info("Progress has been saved. You can resume by running the script again.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
