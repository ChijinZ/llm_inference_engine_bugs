#!/usr/bin/env python3
"""
Script to compare the number of items in JSON files:
- analysis_cache.json
- merged_bug_results.json  
- final_bug_results_with_root_cause_completed_symptom.json
"""

import json
import os
from pathlib import Path

def count_items_in_json(file_path):
    """Count the number of items in a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Special handling for analysis_cache.json
        if file_path.name == "analysis_cache.json" and isinstance(data, dict) and "items" in data:
            return len(data["items"])
        
        if isinstance(data, dict):
            return len(data)
        elif isinstance(data, list):
            return len(data)
        else:
            return 1  # Single item
    except FileNotFoundError:
        return f"File not found: {file_path}"
    except json.JSONDecodeError as e:
        return f"JSON decode error in {file_path}: {e}"
    except Exception as e:
        return f"Error reading {file_path}: {e}"

def main():
    # Define the data folder path
    data_folder = Path("data")
    
    # Define the files to compare
    files_to_compare = [
        "analysis_cache.json",
        "merged_bug_results.json", 
        "final_bug_results_with_root_cause_completed_symptom.json"
    ]
    
    print("JSON File Item Count Comparison")
    print("=" * 50)
    
    results = {}
    
    for filename in files_to_compare:
        file_path = data_folder / filename
        count = count_items_in_json(file_path)
        results[filename] = count
        print(f"{filename}: {count}")
    
    print("\n" + "=" * 50)
    
    # Check if all files were successfully read
    successful_counts = [count for count in results.values() if isinstance(count, int)]
    
    if len(successful_counts) == 3:
        print("Summary:")
        print(f"analysis_cache.json: {results['analysis_cache.json']:,} items")
        print(f"merged_bug_results.json: {results['merged_bug_results.json']:,} items")
        print(f"final_bug_results_with_root_cause_completed_symptom.json: {results['final_bug_results_with_root_cause_completed_symptom.json']:,} items")
        
        # Calculate differences
        cache_count = results['analysis_cache.json']
        merged_count = results['merged_bug_results.json']
        final_count = results['final_bug_results_with_root_cause_completed_symptom.json']
        
        print(f"\nDifferences:")
        print(f"merged_bug_results.json - analysis_cache.json: {merged_count - cache_count:,}")
        print(f"final_bug_results_with_root_cause_completed_symptom.json - merged_bug_results.json: {final_count - merged_count:,}")
        print(f"final_bug_results_with_root_cause_completed_symptom.json - analysis_cache.json: {final_count - cache_count:,}")
        
        # Check for consistency
        if cache_count == merged_count == final_count:
            print("\n✅ All files have the same number of items!")
        else:
            print("\n⚠️  Files have different numbers of items.")
    else:
        print("❌ Some files could not be read successfully. Check the errors above.")

if __name__ == "__main__":
    main()
