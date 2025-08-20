#!/usr/bin/env python3
"""
Script to complete missing root cause analysis for bugs that should have been analyzed but weren't
"""

import json
import logging
import os
from typing import Dict, Any, List, Tuple
from analyze_root_cause import RootCauseAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('complete_missing_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MissingAnalysisCompleter:
    """
    Completes missing root cause analysis for bugs that should have been analyzed
    """
    
    def __init__(self):
        self.analyzer = RootCauseAnalyzer()
    
    def load_current_results(self, filepath: str) -> Dict[str, Any]:
        """Load the current results with root cause analysis"""
        logger.info(f"Loading current results from {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_diff_data(self, filepath: str = 'bug_diffs.json') -> Dict[str, str]:
        """Load diff content data"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Diff file not found: {filepath}")
            return {}
        except Exception as e:
            logger.error(f"Error loading diff file: {e}")
            return {}

    def identify_missing_analyses(self, results: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Identify bugs that should have root cause analysis but don't, including:
        1. Bugs without any root_causes
        2. Bugs with diff content over 5000 characters (may need re-analysis)
        
        Args:
            results: Current results data
            
        Returns:
            List of tuples (url, bug_info) for bugs missing analysis
        """
        missing_analyses = []
        total_fixed_merged = 0
        analyzed_count = 0
        large_diff_count = 0
        
        # Load diff data to check for large diffs
        diff_data = self.load_diff_data()
        
        for url, bug_info in results.items():
            # Check if this should have been analyzed (fixed + merged)
            if (bug_info.get('status') == 'fixed' and 
                bug_info.get('pr_status') == 'Merged'):
                total_fixed_merged += 1
                
                # Check if it has root_causes
                has_root_causes = 'root_causes' in bug_info and bug_info['root_causes']
                
                # Check if diff is large (over 5000 characters)
                has_large_diff = False
                if url in diff_data:
                    diff_content = diff_data[url]
                    if len(diff_content) > 5000:
                        has_large_diff = True
                        large_diff_count += 1
                
                # Add to missing if no root causes OR has large diff
                if not has_root_causes:
                    missing_analyses.append((url, bug_info))
                elif has_large_diff:
                    # Mark as needing re-analysis due to large diff
                    missing_analyses.append((url, bug_info))
                    logger.debug(f"Adding for re-analysis due to large diff ({len(diff_data[url])} chars): {url}")
                else:
                    analyzed_count += 1
        
        logger.info(f"Analysis status:")
        logger.info(f"  Total fixed+merged bugs: {total_fixed_merged}")
        logger.info(f"  Already analyzed (small diffs): {analyzed_count}")
        logger.info(f"  Bugs with large diffs (>5000 chars): {large_diff_count}")
        logger.info(f"  Missing analysis: {len(missing_analyses)}")
        logger.info(f"    - Without root_causes: {len([1 for url, info in missing_analyses if 'root_causes' not in info or not info['root_causes']])}")
        logger.info(f"    - With large diffs needing re-analysis: {len([1 for url, info in missing_analyses if 'root_causes' in info and info['root_causes']])}")
        
        return missing_analyses
    
    def complete_missing_analyses(self, 
                                results: Dict[str, Any],
                                missing_analyses: List[Tuple[str, Dict[str, Any]]],
                                output_file: str,
                                diff_output_file: str = None,
                                max_bugs: int = None) -> Dict[str, Any]:
        """
        Complete the missing root cause analyses
        
        Args:
            results: Current results data
            missing_analyses: List of bugs missing analysis
            output_file: Output file for updated results
            diff_output_file: Optional file for diff content
            max_bugs: Maximum number of bugs to analyze (for testing)
            
        Returns:
            Updated results dictionary
        """
        if max_bugs:
            missing_analyses = missing_analyses[:max_bugs]
            logger.info(f"Limited to {max_bugs} bugs for testing")
        
        # Create copies of the data
        updated_results = results.copy()
        diff_data = {}
        
        # Use the existing caching mechanism from RootCauseAnalyzer
        # Load any existing cache
        cached_results, cached_diff_data, processed_urls = self.analyzer.load_cache()
        
        # Merge cached results into updated_results
        for url, cached_result in cached_results.items():
            if url in updated_results:
                updated_results[url].update(cached_result)
        
        # Merge cached diff data
        diff_data.update(cached_diff_data)
        
        # Filter out already processed bugs
        remaining_analyses = [(url, bug_info) for url, bug_info in missing_analyses if url not in processed_urls]
        
        if processed_urls:
            logger.info(f"Resuming from cache: {len(processed_urls)} bugs already processed, {len(remaining_analyses)} remaining")
        
        # Analyze each remaining missing bug
        analyzed_count = len([url for url in processed_urls if url in dict(missing_analyses)])
        failed_count = 0
        
        for i, (url, bug_info) in enumerate(remaining_analyses):
            logger.info(f"Processing missing analysis {i+1}/{len(remaining_analyses)} (total progress: {len(processed_urls) + i + 1}/{len(missing_analyses)}): {url}")
            
            try:
                # Get diff content
                diff_content = self.analyzer.get_diff_content(url)
                
                # Store diff content if requested
                if diff_output_file and diff_content:
                    diff_data[url] = diff_content
                
                # Analyze root cause if we have diff content
                if diff_content:
                    # Check if this is a re-analysis (already has root_causes)
                    is_reanalysis = 'root_causes' in bug_info and bug_info['root_causes']
                    
                    root_causes = self.analyzer.analyze_root_cause_with_openai(
                        title=bug_info.get('title', ''),
                        description=bug_info.get('description', ''),
                        diff_content=diff_content
                    )
                    
                    if root_causes:
                        # Add/update root_causes in the existing bug entry
                        updated_results[url]['root_causes'] = root_causes
                        analyzed_count += 1
                        if is_reanalysis:
                            logger.info(f"✅ Successfully re-analyzed large diff {url}, found {len(root_causes)} root causes")
                        else:
                            logger.info(f"✅ Successfully analyzed {url}, found {len(root_causes)} root causes")
                    else:
                        failed_count += 1
                        if is_reanalysis:
                            logger.warning(f"❌ No root causes identified for re-analysis of {url}")
                        else:
                            logger.warning(f"❌ No root causes identified for {url}")
                else:
                    failed_count += 1
                    logger.warning(f"❌ No diff content available for {url}")
                    
            except Exception as e:
                failed_count += 1
                logger.error(f"❌ Error analyzing {url}: {e}")
            
            # Mark as processed and save cache
            processed_urls.add(url)
            
            # Save cache every 5 bugs or on successful analysis
            if (i + 1) % 5 == 0 or root_causes:
                self.analyzer.save_cache(updated_results, diff_data, processed_urls)
            
            # Progress logging
            if (i + 1) % 5 == 0:
                logger.info(f"Progress: {i+1}/{len(remaining_analyses)} processed, {analyzed_count} successful, {failed_count} failed")
        
        # Final cache save
        self.analyzer.save_cache(updated_results, diff_data, processed_urls)
        
        # Save updated results
        logger.info(f"Saving updated results to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(updated_results, f, indent=2, ensure_ascii=False)
        
        # Save diff content if requested
        if diff_output_file and diff_data:
            logger.info(f"Saving diff content to {diff_output_file}")
            with open(diff_output_file, 'w', encoding='utf-8') as f:
                json.dump(diff_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(diff_data)} diff contents to {diff_output_file}")
        
        # Clear cache if all analyses are complete
        if len(remaining_analyses) == 0 and analyzed_count + failed_count == len(missing_analyses):
            self.analyzer.clear_cache()
            logger.info("All analyses complete - cache cleared")
        
        logger.info(f"Completion analysis finished!")
        logger.info(f"  Total missing analyses: {len(missing_analyses)}")
        logger.info(f"  Processed this run: {len(remaining_analyses)}")
        logger.info(f"  Successful: {analyzed_count}")
        logger.info(f"  Failed: {failed_count}")
        logger.info(f"  Updated results saved to: {output_file}")
        
        return updated_results


def main():
    """Main function"""
    import sys
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--help':
            print("Usage:")
            print("  python complete_missing_analysis.py          # Complete all missing analyses")
            print("  python complete_missing_analysis.py --test   # Test with first 10 missing")
            print("  python complete_missing_analysis.py --status # Show status only, no analysis")
            print("  python complete_missing_analysis.py --help   # Show this help")
            return 0
        elif sys.argv[1] == '--status':
            # Just show status, don't run analysis
            completer = MissingAnalysisCompleter()
            input_file = 'final_bug_results_with_root_cause.json'
            try:
                current_results = completer.load_current_results(input_file)
                missing_analyses = completer.identify_missing_analyses(current_results)
                if not missing_analyses:
                    print("🎉 No missing analyses found! All fixed+merged bugs already have root cause analysis.")
                else:
                    print(f"📊 Missing Analysis Summary:")
                    print(f"  Total missing: {len(missing_analyses)}")
                    
                    # Categorize missing analyses
                    without_root_causes = [item for item in missing_analyses if 'root_causes' not in item[1] or not item[1]['root_causes']]
                    with_large_diffs = [item for item in missing_analyses if 'root_causes' in item[1] and item[1]['root_causes']]
                    
                    print(f"  - Without root_causes: {len(without_root_causes)}")
                    print(f"  - With large diffs needing re-analysis: {len(with_large_diffs)}")
                    print(f"  First 10 missing bugs:")
                    for i, (url, bug_info) in enumerate(missing_analyses[:10]):
                        has_root_causes = 'root_causes' in bug_info and bug_info['root_causes']
                        status = "re-analysis (large diff)" if has_root_causes else "missing analysis"
                        print(f"    {i+1}. {url} [{status}]")
                        print(f"       Title: {bug_info.get('title', 'N/A')[:80]}...")
                return 0
            except FileNotFoundError:
                print(f"❌ Input file not found: {input_file}")
                return 1
    
    # Check for required environment variables
    if not os.getenv('OPENAI_API_KEY'):
        logger.error("OPENAI_API_KEY environment variable not set")
        return 1

    if not os.getenv('GITHUB_TOKEN'):
        logger.warning("GITHUB_TOKEN environment variable not set - will use unauthenticated requests")

    # Initialize completer
    completer = MissingAnalysisCompleter()
    
    # File paths
    # input_file = 'final_bug_results_with_root_cause.json'
    # output_file = 'final_bug_results_with_root_cause_completed.json'
    input_file = 'final_bug_results_with_root_cause_completed.json'
    output_file = 'final_bug_results_with_root_cause_completed_2.json'
    diff_output_file = 'missing_bug_diffs.json'
    
    try:
        # Load current results
        current_results = completer.load_current_results(input_file)
        
        # Identify missing analyses
        missing_analyses = completer.identify_missing_analyses(current_results)
        
        if not missing_analyses:
            logger.info("🎉 No missing analyses found! All fixed+merged bugs already have root cause analysis.")
            return 0
        
        # Determine if this is a test run
        max_bugs = None
        if len(sys.argv) > 1 and sys.argv[1] == '--test':
            max_bugs = 10
            logger.info("Running in test mode - limited to 10 bugs")
        
        # Complete missing analyses
        updated_results = completer.complete_missing_analyses(
            current_results, 
            missing_analyses, 
            output_file, 
            diff_output_file,
            max_bugs
        )
        
        return 0
        
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_file}")
        logger.error("Make sure you have run the main analysis first to generate the input file.")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
