#!/usr/bin/env python3
"""
Example usage of the LLM Bug Collector

This script demonstrates how to use the LLMBugCollector class programmatically
with custom configurations and filtering options.
"""

import os
from collect_issues import LLMBugCollector, BugStatus, IssueType

def example_basic_usage():
    """Basic usage example"""
    print("=== Basic Usage Example ===")
    
    # Get API keys from environment
    github_token = os.getenv('GITHUB_TOKEN')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    if not openai_api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    if not github_token:
        print("⚠️  Warning: GITHUB_TOKEN not set - will use unauthenticated API (60 requests/hour limit)")
    
    # Create collector
    collector = LLMBugCollector(openai_api_key, github_token)
    
    # Collect bugs from all repositories
    bugs = collector.collect_bugs()
    
    # Save results
    collector.save_results(bugs, "data/example_bugs.json")
    
    # Print summary
    collector.print_summary(bugs)
    
    return bugs

def example_custom_repositories():
    """Example with custom repository list"""
    print("\n=== Custom Repositories Example ===")
    
    github_token = os.getenv('GITHUB_TOKEN')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    if not github_token or not openai_api_key:
        print("Please set GITHUB_TOKEN and OPENAI_API_KEY environment variables")
        return
    
    # Create collector
    collector = LLMBugCollector(github_token, openai_api_key)
    
    # Override repositories to analyze only specific ones
    collector.repositories = [
        "ggerganov/llama.cpp",
        "vllm-project/vllm"
    ]
    
    print(f"Analyzing repositories: {collector.repositories}")
    
    # Collect bugs
    bugs = collector.collect_bugs()
    
    # Save results
    collector.save_results(bugs, "data/custom_repos_bugs.json")
    
    return bugs

def example_filter_by_status(bugs):
    """Example of filtering bugs by status"""
    print("\n=== Filtering by Status Example ===")
    
    # Filter for only fixed bugs
    fixed_bugs = [bug for bug in bugs if bug.status == BugStatus.FIXED]
    print(f"Fixed bugs: {len(fixed_bugs)}")
    
    # Filter for only confirmed but not fixed bugs
    confirmed_bugs = [bug for bug in bugs if bug.status == BugStatus.CONFIRMED_NOT_FIXED]
    print(f"Confirmed but not fixed: {len(confirmed_bugs)}")
    
    # Filter for only issues (not PRs)
    issues_only = [bug for bug in bugs if bug.type == IssueType.ISSUE]
    print(f"Issues only: {len(issues_only)}")
    
    return {
        'fixed': fixed_bugs,
        'confirmed': confirmed_bugs,
        'issues': issues_only
    }

def example_repository_analysis(bugs):
    """Example of analyzing bugs by repository"""
    print("\n=== Repository Analysis Example ===")
    
    # Group bugs by repository
    repo_bugs = {}
    for bug in bugs:
        if bug.repository not in repo_bugs:
            repo_bugs[bug.repository] = []
        repo_bugs[bug.repository].append(bug)
    
    # Analyze each repository
    for repo, repo_bug_list in repo_bugs.items():
        print(f"\n{repo}:")
        print(f"  Total bugs: {len(repo_bug_list)}")
        
        # Count by status
        status_counts = {}
        for bug in repo_bug_list:
            status = bug.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        for status, count in status_counts.items():
            print(f"  {status}: {count}")
    
    return repo_bugs

def example_export_filtered_results(bugs, filename="data/filtered_bugs.json"):
    """Example of exporting filtered results"""
    print(f"\n=== Exporting Filtered Results to {filename} ===")
    
    # Filter for high-priority bugs (fixed or confirmed)
    high_priority = [
        bug for bug in bugs 
        if bug.status in [BugStatus.FIXED, BugStatus.CONFIRMED_NOT_FIXED]
    ]
    
    # Create filtered results
    filtered_results = {
        'metadata': {
            'filter': 'high_priority_bugs',
            'total_bugs': len(high_priority),
            'description': 'Bugs that are either fixed or confirmed but not fixed'
        },
        'bugs': [bug.to_dict() for bug in high_priority]
    }
    
    # Save to file
    import json
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(filtered_results, f, indent=2, ensure_ascii=False)
    
    print(f"Exported {len(high_priority)} high-priority bugs to {filename}")

def main():
    """Run all examples"""
    print("LLM Bug Collector - Example Usage\n")
    
    # Check if API keys are available
    github_token = os.getenv('GITHUB_TOKEN')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    if not github_token or not openai_api_key:
        print("⚠️  API keys not found. Please set:")
        print("   export GITHUB_TOKEN='your_github_token'")
        print("   export OPENAI_API_KEY='your_openai_key'")
        print("\nSkipping actual API calls and showing example structure...")
        
        # Show example data structure
        from collect_issues import BugReport, BugStatus, IssueType
        
        example_bug = BugReport(
            url="https://github.com/example/repo/issues/123",
            type=IssueType.ISSUE,
            status=BugStatus.FIXED,
            date="2024-01-15T10:30:00Z",
            fix_url="https://github.com/example/repo/pull/456",
            description="Example bug description",
            title="Example Bug Title",
            repository="example/repo",
            number=123,
            labels=["bug", "high-priority"],
            assignees=["developer1"],
            milestone="v1.2.0"
        )
        
        print("\nExample Bug Report Structure:")
        print(json.dumps(example_bug.to_dict(), indent=2))
        return
    
    try:
        # Run examples
        bugs = example_basic_usage()
        
        if bugs:
            # Run filtering examples
            filtered = example_filter_by_status(bugs)
            repo_analysis = example_repository_analysis(bugs)
            example_export_filtered_results(bugs)
        
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Check your API keys and network connection.")

if __name__ == "__main__":
    main()
