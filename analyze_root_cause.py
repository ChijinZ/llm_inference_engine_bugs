#!/usr/bin/env python3
"""
Script to analyze root causes of fixed bugs using OpenAI GPT analysis

This script:
1. Reads final_bug_results.json
2. Filters bugs with status="fixed" and pr_status="Merged"
3. Fetches PR diff content using GitHubAPIClient
4. Uses OpenAI GPT-5-nano to analyze title, description, and diff
5. Extracts root cause locations (file paths and functions)
6. Saves results in JSON format

Usage:
    python analyze_root_cause.py

Requirements:
    - OPENAI_API_KEY environment variable
    - GITHUB_TOKEN environment variable (recommended)
    - final_bug_results.json in current directory

Output:
    - root_cause_analysis.json: Analysis results
    - root_cause_analysis.log: Detailed logs
"""

import json
import logging
import os
import time
import pickle
from typing import List, Dict, Any, Optional
from openai import OpenAI
from common import GitHubAPIClient

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler('root_cause_analysis.log'),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# OpenAI prompt template for root cause analysis
ROOT_CAUSE_ANALYSIS_PROMPT = """You are a software engineering expert analyzing a bug fix. Based on the bug title, summary, and the actual code diff that fixed the issue, identify the root cause locations.

BUG TITLE: {title}

BUG SUMMARY: {description}

CODE DIFF:
{diff_content}

Please analyze the diff and identify the most important root cause locations (maximum 3) that really matter for this issue. Focus on:
1. Files and functions that contain the core logic changes
2. Areas where the actual bug was located (not just cosmetic changes)
3. The most critical functions that were modified to fix the root cause

Return ONLY a valid JSON object with a "root_causes" array containing file paths and function names, like this format:
{{"root_causes": [{{"file": "path/to/file.cpp", "function": "function_name"}}, {{"file": "another/file.h", "function": "another_function"}}]}}

Important:
- Return maximum 3 entries in the root_causes array
- Focus on the most important root causes only
- Use exact file paths from the diff
- Use exact function names from the diff
- Return only the JSON object with root_causes key, no other text"""


class RootCauseAnalyzer:
    """
    Analyzes root causes of fixed bugs using OpenAI GPT analysis
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the analyzer with OpenAI and GitHub clients
        
        Args:
            openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        # Initialize OpenAI client
        self.openai_client = OpenAI(
            api_key=openai_api_key or os.getenv('OPENAI_API_KEY'))

        # Initialize GitHub client
        self.github_client = GitHubAPIClient()

        # Rate limiting for OpenAI API
        self.openai_delay = 1.0  # 1 second between OpenAI calls
        
        # Cache file paths
        self.cache_dir = 'cache'
        self.progress_cache_file = os.path.join(self.cache_dir, 'analysis_progress.pkl')
        self.results_cache_file = os.path.join(self.cache_dir, 'analysis_results.pkl')
        self.diff_cache_file = os.path.join(self.cache_dir, 'diff_cache.pkl')
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

    def save_cache(self, results: Dict[str, Any], diff_data: Dict[str, str], processed_urls: set):
        """Save current progress to cache files"""
        try:
            # Save results
            with open(self.results_cache_file, 'wb') as f:
                pickle.dump(results, f)
            
            # Save diff data
            with open(self.diff_cache_file, 'wb') as f:
                pickle.dump(diff_data, f)
            
            # Save progress (processed URLs)
            with open(self.progress_cache_file, 'wb') as f:
                pickle.dump(processed_urls, f)
            
            logger.info(f"Cache saved: {len(processed_urls)} bugs processed")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def load_cache(self) -> tuple[Dict[str, Any], Dict[str, str], set]:
        """Load cached progress and results"""
        try:
            # Load results
            if os.path.exists(self.results_cache_file):
                with open(self.results_cache_file, 'rb') as f:
                    results = pickle.load(f)
            else:
                results = {}
            
            # Load diff data
            if os.path.exists(self.diff_cache_file):
                with open(self.diff_cache_file, 'rb') as f:
                    diff_data = pickle.load(f)
            else:
                diff_data = {}
            
            # Load progress
            if os.path.exists(self.progress_cache_file):
                with open(self.progress_cache_file, 'rb') as f:
                    processed_urls = pickle.load(f)
            else:
                processed_urls = set()
            
            logger.info(f"Cache loaded: {len(processed_urls)} bugs already processed")
            return results, diff_data, processed_urls
            
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return {}, {}, set()

    def clear_cache(self):
        """Clear all cache files"""
        try:
            for cache_file in [self.results_cache_file, self.diff_cache_file, self.progress_cache_file]:
                if os.path.exists(cache_file):
                    os.remove(cache_file)
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")

    def load_bug_data(self, filepath: str) -> Dict[str, Any]:
        """Load bug data from JSON file"""
        logger.info(f"Loading bug data from {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def filter_fixed_merged_bugs(self, bug_data: Dict[str,
                                                      Any]) -> List[tuple]:
        """
        Filter bugs that are fixed and have merged PR status
        
        Returns:
            List of tuples (url, bug_info)
        """
        fixed_merged_bugs = []

        for url, bug_info in bug_data.items():
            if (bug_info.get('status') == 'fixed'
                    and bug_info.get('pr_status') == 'Merged'):
                fixed_merged_bugs.append((url, bug_info))

        logger.info(f"Found {len(fixed_merged_bugs)} fixed and merged bugs")
        return fixed_merged_bugs

    def analyze_root_cause_with_openai(
            self, title: str, description: str,
            diff_content: str) -> List[Dict[str, str]]:
        """
        Use OpenAI to analyze root cause from title, description, and diff
        
        Args:
            title: Bug title
            description: Bug description
            diff_content: PR diff content
            
        Returns:
            List of dictionaries with file and function information
        """
        prompt = ROOT_CAUSE_ANALYSIS_PROMPT.format(title=title,
                                                   description=description,
                                                   diff_content=diff_content)

        try:
            # Rate limiting
            time.sleep(self.openai_delay)

            response = self.openai_client.chat.completions.create(
                model="gpt-5-nano",  # As requested by user
                messages=[{
                    "role":
                    "system",
                    "content":
                    "You are a software engineering expert who analyzes code diffs to identify root causes of bugs. Always return valid JSON objects with a 'root_causes' array."
                }, {
                    "role": "user",
                    "content": prompt
                }],
                response_format={"type": "json_object"})

            result_text = response.choices[0].message.content.strip()
            logger.debug(f"OpenAI response: {result_text}")

                        # Try to parse the JSON response
            try:
                response_data = json.loads(result_text)
                
                # Extract root_causes array from the JSON object
                if isinstance(response_data, dict) and 'root_causes' in response_data:
                    root_causes = response_data['root_causes']
                    
                    # Validate the format
                    if isinstance(root_causes, list) and len(root_causes) <= 3:
                        for item in root_causes:
                            if not isinstance(item, dict) or 'file' not in item or 'function' not in item:
                                logger.warning(f"Invalid root cause format: {item}")
                                return []
                        return root_causes
                    else:
                        logger.warning(f"Invalid root_causes array format or too many entries: {root_causes}")
                        return []
                else:
                    logger.warning(f"Expected JSON object with 'root_causes' key, got: {response_data}")
                    return []

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse OpenAI response as JSON: {e}")
                logger.error(f"Response was: {result_text}")
                return []

        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return []

    def run_analysis(self,
                     input_file: str,
                     output_file: str,
                     diff_output_file: str = None,
                     max_bugs: Optional[int] = None):
        """
        Run the complete root cause analysis
        
        Args:
            input_file: Path to final_bug_results.json
            output_file: Path to save analysis results (same format as input)
            diff_output_file: Path to save diff content ({"url": "content"} format)
            max_bugs: Maximum number of bugs to analyze (for testing)
        """
        logger.info("Starting root cause analysis")

        # Load bug data
        bug_data = self.load_bug_data(input_file)
        
        # Load cached progress
        cached_results, cached_diff_data, processed_urls = self.load_cache()
        
        # Create a copy of the original data to preserve structure
        result_data = bug_data.copy()
        
        # Restore cached results
        for url, cached_result in cached_results.items():
            if url in result_data:
                result_data[url].update(cached_result)
        
        # Dictionary to store diff content
        diff_data = cached_diff_data.copy()

        # Filter fixed and merged bugs
        fixed_merged_bugs = self.filter_fixed_merged_bugs(bug_data)
        
        # Skip already processed bugs
        remaining_bugs = [(url, bug_info) for url, bug_info in fixed_merged_bugs if url not in processed_urls]
        
        if processed_urls:
            logger.info(f"Resuming from cache: {len(processed_urls)} bugs already processed, {len(remaining_bugs)} remaining")
        else:
            logger.info(f"Starting fresh analysis: {len(remaining_bugs)} bugs to process")

        if max_bugs:
            remaining_bugs = remaining_bugs[:max_bugs]
            logger.info(f"Limited analysis to {max_bugs} bugs for testing")

        # Analyze each bug and add root_causes to the original structure
        analyzed_count = len(processed_urls)  # Start with already processed count
        failed_count = 0

        for i, (url, bug_info) in enumerate(remaining_bugs):
            logger.info(f"Processing bug {i+1}/{len(remaining_bugs)} (total progress: {len(processed_urls) + i + 1}/{len(fixed_merged_bugs)})")

            # Get diff content
            diff_content = self.get_diff_content(url)
            
            # Store diff content if requested
            if diff_output_file and diff_content:
                diff_data[url] = diff_content
            
            # Analyze root cause if we have diff content
            if diff_content:
                root_causes = self.analyze_root_cause_with_openai(
                    title=bug_info.get('title', ''),
                    description=bug_info.get('description', ''),
                    diff_content=diff_content
                )
                
                if root_causes:
                    # Add root_causes to the existing bug entry
                    result_data[url]['root_causes'] = root_causes
                    analyzed_count += 1
                    logger.info(f"Successfully analyzed {url}, found {len(root_causes)} root causes")
                else:
                    failed_count += 1
                    logger.warning(f"No root causes identified for {url}")
            else:
                failed_count += 1
                logger.warning(f"No diff content available for {url}")
            
            # Mark as processed and save cache
            processed_urls.add(url)
            
            # Save cache every 10 bugs or on successful analysis
            if (i + 1) % 10 == 0 or (root_causes and len(root_causes) > 0):
                self.save_cache(result_data, diff_data, processed_urls)

            # Progress logging
            if (i + 1) % 10 == 0:
                logger.info(
                    f"Processed {i+1}/{len(remaining_bugs)} bugs, {analyzed_count} successful, {failed_count} failed"
                )

        # Final cache save
        self.save_cache(result_data, diff_data, processed_urls)
        
        # Save results in the same format as input
        logger.info(f"Saving results to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        # Save diff content if requested
        if diff_output_file and diff_data:
            logger.info(f"Saving diff content to {diff_output_file}")
            with open(diff_output_file, 'w', encoding='utf-8') as f:
                json.dump(diff_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(diff_data)} diff contents to {diff_output_file}")

        logger.info(
            f"Analysis complete! Processed {len(fixed_merged_bugs)} bugs, {analyzed_count} successful, {failed_count} failed"
        )
        logger.info(f"Results saved to {output_file}")
        if diff_output_file and diff_data:
            logger.info(f"Diff content saved to {diff_output_file}")
        
        # Clear cache after successful completion
        self.clear_cache()
        logger.info("Cache cleared after successful completion")
    
    def get_diff_content(self, url: str) -> str:
        """
        Get diff content for a PR URL with error handling
        
        Args:
            url: PR URL
            
        Returns:
            Diff content or empty string if failed
        """
        try:
            # logger.info(f"Fetching diff for {url}")
            diff_content = self.github_client.get_pull_request_diff(url)
            
            if not diff_content or len(diff_content.strip()) == 0:
                logger.warning(f"No diff content found for {url}")
                return ""
            
            logger.info(f"Diff content length: {len(diff_content)} characters")
            
            # Truncate diff if too long (OpenAI has token limits)
            max_diff_length = 50000  # Reasonable limit for GPT analysis
            if len(diff_content) > max_diff_length:
                logger.warning(
                    f"Diff too long ({len(diff_content)} chars), truncating to {max_diff_length}"
                )
                diff_content = diff_content[:max_diff_length] + "\n\n... (truncated)"
            
            return diff_content
            
        except Exception as e:
            logger.error(f"Error fetching diff for {url}: {e}")
            return ""


def main():
    """Main function"""
    import sys
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--clear-cache':
            analyzer = RootCauseAnalyzer()
            analyzer.clear_cache()
            print("Cache cleared successfully")
            return 0
        elif sys.argv[1] == '--help':
            print("Usage:")
            print("  python analyze_root_cause.py          # Run analysis")
            print("  python analyze_root_cause.py --clear-cache  # Clear cache")
            print("  python analyze_root_cause.py --help   # Show this help")
            return 0
    
    # Check for required environment variables
    if not os.getenv('OPENAI_API_KEY'):
        logger.error("OPENAI_API_KEY environment variable not set")
        return 1

    if not os.getenv('GITHUB_TOKEN'):
        logger.warning(
            "GITHUB_TOKEN environment variable not set - will use unauthenticated requests"
        )

    # Initialize analyzer
    analyzer = RootCauseAnalyzer()

    # Run analysis
    input_file = 'final_bug_results.json'
    output_file = 'final_bug_results_with_root_cause.json'
    diff_output_file = 'bug_diffs.json'

    # For testing, uncomment the line below to limit to first 5 bugs
    # analyzer.run_analysis(input_file, output_file, diff_output_file, max_bugs=5)

    # Full analysis (comment out for testing)
    analyzer.run_analysis(input_file, output_file, diff_output_file)

    return 0


if __name__ == "__main__":
    exit(main())
