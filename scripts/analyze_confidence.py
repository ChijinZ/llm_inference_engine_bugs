#!/usr/bin/env python3
"""
Analyze confidence distribution of fixed bugs from merged_bug_results.json

This script reads the merged bug results and analyzes the distribution of
confidence scores for bugs that are marked as fixed.
"""

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConfidenceAnalyzer:
    """Analyze confidence distribution of fixed bugs"""
    
    def __init__(self, merged_results_file: str):
        self.merged_results_file = merged_results_file
        self.data = self.load_json_file(merged_results_file)
        
    def load_json_file(self, file_path: str) -> Dict[str, Any]:
        """Load JSON data from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return {}
    
    def filter_fixed_bugs(self) -> List[Dict[str, Any]]:
        """Filter bugs that are marked as fixed"""
        fixed_bugs = []
        
        for url, bug_data in self.data.items():
            status = bug_data.get('status', '').lower()
            if 'fixed' in status or 'closed' in status or 'resolved' in status:
                fixed_bugs.append({
                    'url': url,
                    'confidence': bug_data.get('confidence_for_bug_related'),
                    'is_bug_related': bug_data.get('is_bug_related'),
                    'title': bug_data.get('title', ''),
                    'status': bug_data.get('status', '')
                })
        
        return fixed_bugs
    
    def analyze_confidence_distribution(self, fixed_bugs: List[Dict[str, Any]]):
        """Analyze and print confidence distribution"""
        if not fixed_bugs:
            logger.warning("No fixed bugs found!")
            return
        
        # Filter bugs with valid confidence scores
        bugs_with_confidence = [bug for bug in fixed_bugs if bug['confidence'] is not None]
        bugs_without_confidence = [bug for bug in fixed_bugs if bug['confidence'] is None]
        
        logger.info(f"Total fixed bugs: {len(fixed_bugs)}")
        logger.info(f"Bugs with confidence scores: {len(bugs_with_confidence)}")
        logger.info(f"Bugs without confidence scores: {len(bugs_without_confidence)}")
        
        if not bugs_with_confidence:
            logger.warning("No bugs with confidence scores found!")
            return
        
        # Extract confidence scores
        confidence_scores = [bug['confidence'] for bug in bugs_with_confidence]
        
        # Basic statistics
        min_conf = min(confidence_scores)
        max_conf = max(confidence_scores)
        avg_conf = sum(confidence_scores) / len(confidence_scores)
        
        print("\n" + "="*60)
        print("CONFIDENCE DISTRIBUTION ANALYSIS")
        print("="*60)
        print(f"Total fixed bugs analyzed: {len(fixed_bugs)}")
        print(f"Bugs with confidence scores: {len(bugs_with_confidence)}")
        print(f"Bugs without confidence scores: {len(bugs_without_confidence)}")
        print()
        
        print("CONFIDENCE SCORE STATISTICS:")
        print(f"  Minimum confidence: {min_conf:.3f}")
        print(f"  Maximum confidence: {max_conf:.3f}")
        print(f"  Average confidence: {avg_conf:.3f}")
        print()
        
        # Create confidence ranges
        ranges = [
            (0.0, 0.6, "Low (0.0-0.6)"),
            (0.6, 0.85, "Medium (0.6-0.85)"),
            (0.85, 0.9, "High (0.85-0.9)"),
            (0.9, 0.95, "Very High (0.9-0.95)"),
            (0.95, 1.0, "Extremely High (0.95-1.0)")
        ]
        
        print("CONFIDENCE RANGE DISTRIBUTION:")
        print("-" * 40)
        
        for min_val, max_val, label in ranges:
            count = sum(1 for conf in confidence_scores if min_val <= conf < max_val)
            percentage = (count / len(confidence_scores)) * 100
            print(f"{label:20} | {count:4d} bugs | {percentage:5.1f}%")
        
        # Special case for exactly 1.0
        count_1_0 = sum(1 for conf in confidence_scores if conf == 1.0)
        if count_1_0 > 0:
            percentage = (count_1_0 / len(confidence_scores)) * 100
            print(f"{'Perfect (1.0)':20} | {count_1_0:4d} bugs | {percentage:5.1f}%")
        
        print()
        
        # Bug-related vs not bug-related analysis
        bug_related = [bug for bug in bugs_with_confidence if bug['is_bug_related'] is True]
        not_bug_related = [bug for bug in bugs_with_confidence if bug['is_bug_related'] is False]
        unknown = [bug for bug in bugs_with_confidence if bug['is_bug_related'] is None]
        
        print("CONFIDENCE BY BUG CLASSIFICATION:")
        print("-" * 40)
        
        if bug_related:
            avg_conf_bug = sum(bug['confidence'] for bug in bug_related) / len(bug_related)
            print(f"Bug-related bugs:     {len(bug_related):4d} bugs | Avg confidence: {avg_conf_bug:.3f}")
        
        if not_bug_related:
            avg_conf_not = sum(bug['confidence'] for bug in not_bug_related) / len(not_bug_related)
            print(f"Not bug-related bugs: {len(not_bug_related):4d} bugs | Avg confidence: {avg_conf_not:.3f}")
        
        if unknown:
            avg_conf_unknown = sum(bug['confidence'] for bug in unknown) / len(unknown)
            print(f"Unknown classification: {len(unknown):4d} bugs | Avg confidence: {avg_conf_unknown:.3f}")
        
        print()
        
        # Show examples of high and low confidence bugs
        print("EXAMPLES OF HIGH CONFIDENCE BUGS (>0.9):")
        print("-" * 50)
        high_conf_bugs = [bug for bug in bugs_with_confidence if bug['confidence'] > 0.9]
        for i, bug in enumerate(high_conf_bugs[:5], 1):  # Show first 5
            print(f"{i}. Confidence: {bug['confidence']:.3f} | {bug['title'][:60]}...")
            print(f"   URL: {bug['url']}")
            print()
        
        print("EXAMPLES OF LOW CONFIDENCE BUGS (<0.3):")
        print("-" * 50)
        low_conf_bugs = [bug for bug in bugs_with_confidence if bug['confidence'] < 0.3]
        for i, bug in enumerate(low_conf_bugs[:5], 1):  # Show first 5
            print(f"{i}. Confidence: {bug['confidence']:.3f} | {bug['title'][:60]}...")
            print(f"   URL: {bug['url']}")
            print()
    
    def run(self):
        """Main analysis function"""
        logger.info("Starting confidence distribution analysis...")
        
        if not self.data:
            logger.error("No data loaded!")
            return
        
        logger.info(f"Loaded {len(self.data)} bug results from {self.merged_results_file}")
        
        # Filter fixed bugs
        fixed_bugs = self.filter_fixed_bugs()
        logger.info(f"Found {len(fixed_bugs)} fixed bugs")
        
        # Analyze confidence distribution
        self.analyze_confidence_distribution(fixed_bugs)
        
        logger.info("Confidence analysis completed!")


def main():
    """Main function"""
    merged_results_file = "data/merged_bug_results.json"
    
    if not Path(merged_results_file).exists():
        logger.error(f"File not found: {merged_results_file}")
        sys.exit(1)
    
    analyzer = ConfidenceAnalyzer(merged_results_file)
    analyzer.run()


if __name__ == "__main__":
    main()
