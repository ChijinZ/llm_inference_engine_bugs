#!/usr/bin/env python3
"""
Script to add PR status information to merged bug results
"""

import json
import logging
from urllib.parse import urlparse
from common import GitHubAPIClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('add_pr_status.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def is_github_pull_request_url(url: str) -> bool:
    """Check if a URL is a GitHub pull request URL"""
    try:
        parsed_url = urlparse(url)
        if parsed_url.hostname != 'github.com':
            return False
        
        path_parts = parsed_url.path.strip('/').split('/')
        return (len(path_parts) >= 4 and path_parts[2] == 'pull')
    except Exception:
        return False


def process_bug_results():
    """Process merged bug results and add PR status information"""
    
    # Initialize GitHub API client
    github_client = GitHubAPIClient()
    
    # Load merged bug results
    logger.info("Loading merged bug results...")
    try:
        with open('merged_bug_results.json', 'r', encoding='utf-8') as f:
            bug_results = json.load(f)
    except FileNotFoundError:
        logger.error("merged_bug_results.json not found")
        return
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing merged_bug_results.json: {e}")
        return
    
    total_items = len(bug_results)
    logger.info(f"Processing {total_items} items...")
    
    processed_count = 0
    pr_status_added_count = 0
    error_count = 0
    
    # Process each item
    for url, item_data in bug_results.items():
        processed_count += 1
        
        # Log progress every 100 items
        if processed_count % 100 == 0:
            logger.info(f"Progress: {processed_count}/{total_items} ({processed_count/total_items*100:.1f}%)")
        
        # Initialize pr_status as null
        item_data['pr_status'] = None
        
        try:
            # Check if item status is "fixed" and URL is a GitHub PR
            item_status = item_data.get('status', '')
            
            if item_status == 'fixed' and is_github_pull_request_url(url):
                logger.debug(f"Fetching PR status for: {url}")
                
                # Get PR status using GitHubAPIClient
                try:
                    pr_status = github_client.get_pull_request_status(url)
                    item_data['pr_status'] = pr_status
                    pr_status_added_count += 1
                    logger.debug(f"PR status for {url}: {pr_status}")
                    
                except ValueError as e:
                    logger.warning(f"Invalid URL {url}: {e}")
                    error_count += 1
                    
                except RuntimeError as e:
                    logger.error(f"Failed to fetch PR status for {url}: {e}")
                    error_count += 1
                    
                except Exception as e:
                    logger.error(f"Unexpected error fetching PR status for {url}: {e}")
                    error_count += 1
            else:
                logger.debug(f"Skipping {url}: status='{item_status}', is_pr={is_github_pull_request_url(url)}")
                
        except Exception as e:
            logger.error(f"Error processing item {url}: {e}")
            error_count += 1
    
    # Save results to final_bug_results.json
    logger.info("Saving results to final_bug_results.json...")
    try:
        with open('final_bug_results.json', 'w', encoding='utf-8') as f:
            json.dump(bug_results, f, indent=2, ensure_ascii=False)
        logger.info("Results saved successfully")
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return
    
    # Print summary
    logger.info(f"""
Processing completed:
- Total items processed: {processed_count}
- PR status added: {pr_status_added_count}
- Errors encountered: {error_count}
- Results saved to: final_bug_results.json
""")


if __name__ == "__main__":
    process_bug_results()
