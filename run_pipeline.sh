#!/bin/bash
# LLM Bug Analysis Pipeline Runner
# 
# This script runs the complete bug analysis pipeline from the root directory.
# Make sure you have set up your environment variables:
#   export OPENAI_API_KEY="your-api-key"
#   export GITHUB_TOKEN="your-github-token"

set -e  # Exit on any error

echo "🔍 Starting LLM Bug Analysis Pipeline..."
echo "📁 Working directory: $(pwd)"
echo "📋 Data will be saved to: data/"
echo ""

# Check for required environment variables
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY not set. Root cause analysis will fail."
fi

if [ -z "$GITHUB_TOKEN" ]; then
    echo "⚠️  Warning: GITHUB_TOKEN not set. API rate limits will be lower."
fi

echo ""
echo "📥 Step 1: Collecting bugs from GitHub repositories..."
python -m scripts.collect_issues

echo ""
echo "🤖 Step 2: Classifying issues using LLM..."
python -m scripts.classifier

echo ""
echo "🔗 Step 3: Merging collection and classification results..."
python -m scripts.merge_bug_results

echo ""
echo "📊 Step 4: Adding PR status information..."
python -m scripts.add_pr_status

echo ""
echo "🧠 Step 5: Analyzing root causes of fixed bugs..."
python -m scripts.analyze_root_cause

echo ""
echo "✅ Step 6: Completing missing analyses..."
python -m scripts.complete_missing_analysis

echo ""
echo "🎉 Pipeline completed successfully!"
echo "📁 Check the data/ folder for results:"
echo "   - data/llm_bugs.json (raw data)"
echo "   - data/final_bug_results_with_root_cause_completed.json (final results)"
echo "   - data/*.log (detailed logs)"
