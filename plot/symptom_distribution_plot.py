#!/usr/bin/env python3
"""
Script to plot symptom distribution by project.
Creates a percentage-based stacked bar chart showing symptom distribution for each project.

For Chinese language support:
1. Install Chinese fonts: sudo apt install fonts-noto-cjk fonts-wqy-microhei
2. Set LANGUAGE = "zh" in the configuration below
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import re
from pathlib import Path
import matplotlib.colors as mc

# Static color configuration for symptoms (medium-toned, distinguishable palette)
SYMPTOM_COLORS = {
    "Incorrect/Inaccuracy Output": "#FF6B6B",        # Medium Red
    "Build/Compilation Error": "#4ECDC4",           # Medium Teal
    "Runtime Crash": "#FFB347",                     # Medium Orange
    "Performance/Memory Issue": "#98D8C8",          # Medium Mint Green
    "API/argument parsing Error": "#F7DC6F",        # Medium Yellow
    "Hardware/Backend Compatibility Issue": "#BB8FCE",  # Medium Purple
    "Model Loading Error": "#85C1E9",               # Medium Sky Blue
    "Distributed Parallelism/Sharding Bug": "#F8BBD9",  # Medium Pink
    "Security Vulnerability": "cyan",            
    "Other": "#BDC3C7"                              # Medium Gray
}

# Language configuration
LANGUAGE = "en"  # Change to "zh" for Chinese version (requires Chinese fonts)

# Text labels based on language
LABELS = {
    "en": {
        "percentage": "Percentage (%)",
        "symptom": "Bug Symptom",
        "symptom_summary": "BUG SYMPTOM DISTRIBUTION BY PROJECT"
    },
    "zh": {
        "percentage": "百分比 (%)",
        "symptom": "Bug症状",
        "symptom_summary": "按项目的Bug症状分布"
    }
}


def extract_project_name(url):
    """Extract simplified project name from GitHub URL."""
    # Extract org/repo from URL like https://github.com/ggml-org/llama.cpp/pull/15304
    match = re.search(r'github\.com/([^/]+/[^/]+)', url)
    if match:
        org_repo = match.group(1)
        # Simplify project names
        if org_repo == "vllm-project/vllm":
            return "vllm"
        elif org_repo == "ggml-org/llama.cpp":
            return "llama.cpp"
        elif org_repo == "deepspeedai/DeepSpeed":
            return "DeepSpeed"
        elif org_repo == "NVIDIA/TensorRT-LLM":
            return "TensorRT-LLM"
        elif org_repo == "huggingface/text-generation-inference":
            return "TGI"
        elif org_repo == "mlc-ai/mlc-llm":
            return "mlc-llm"
        else:
            # For any other projects, extract just the repo name
            repo_name = org_repo.split('/')[-1]
            return repo_name
    return "Unknown"


def normalize_symptom(symptom):
    """Normalize symptom names and group less common ones into 'Other'."""
    assert symptom, "Symptom is empty"
    # if not symptom:
    #     return "Other"
    
    # Group all "other:" symptoms into "Other"
    if symptom.startswith("other:"):
        return "Other"
    
    # Keep main symptom categories (consolidated for academic presentation)
    main_symptoms = {
        "Incorrect/Inaccuracy Output",
        "Build/Compilation Error", 
        "Runtime Crash",
        "Performance/Memory Issue",
        "API/argument parsing Error",
        "Hardware/Backend Compatibility Issue",
        "Model Loading Error",
        "Distributed Parallelism/Sharding Bug",
        "Security Vulnerability",
    }
    if symptom == "Security vulnerability":
        return "Security Vulnerability"
    
    if symptom in main_symptoms:
        return symptom
    else:
        return "Other"


def load_and_process_data(file_path):
    """Load data and process it by project and symptom."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Filter out bugs where is_bug_related is False and exclude Feature Request
    bug_related_data = {
        k: v
        for k, v in data.items() if v.get("is_bug_related") == True
        and v.get("status") == "fixed"
        and v.get("pr_status") == "Merged"
        and v.get("symptom") != "Feature Request"
    }

    # Group by project and symptom
    project_symptom_stats = defaultdict(lambda: defaultdict(int))
    
    for url, bug_data in bug_related_data.items():
        project_name = extract_project_name(url)
        symptom = normalize_symptom(bug_data.get("symptom", ""))
        project_symptom_stats[project_name][symptom] += 1

    return project_symptom_stats


def create_symptom_distribution_plot(project_symptom_stats, output_path="symptom_distribution_by_project.png"):
    """Create pie charts showing symptom distribution for each project in a 3x2 subplot layout."""
    # Define the fixed order of projects
    project_order = [
        "llama.cpp", "vllm", "DeepSpeed", "mlc-llm", "TensorRT-LLM", "TGI"
    ]

    # Filter projects to only include those in our defined order
    projects = [
        project for project in project_order
        if project in project_symptom_stats.keys()
    ]

    # Get all symptoms that appear in the data
    all_symptoms = set()
    for project_stats in project_symptom_stats.values():
        all_symptoms.update(project_stats.keys())
    
    # Sort symptoms by total frequency (descending)
    symptom_totals = defaultdict(int)
    for project_stats in project_symptom_stats.values():
        for symptom, count in project_stats.items():
            symptom_totals[symptom] += count
    
    symptoms = sorted(all_symptoms, key=lambda x: symptom_totals[x], reverse=True)
    
    # Get colors for symptoms
    symptom_color_map = {}
    for i, symptom in enumerate(symptoms):
        symptom_color_map[symptom] = SYMPTOM_COLORS.get(symptom, "#BDC3C7")

    # Create the plot with academic styling
    if LANGUAGE == "zh":
        # For Chinese, use a simpler approach that should work better
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Noto Sans CJK JP'],
            'font.size': 10,
            'axes.linewidth': 0.5,
            'axes.spines.left': False,
            'axes.spines.bottom': False,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.unicode_minus': False
        })
    else:
        # Use serif fonts for English academic style
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
            'font.size': 10,
            'axes.linewidth': 0.5,
            'axes.spines.left': False,
            'axes.spines.bottom': False,
            'axes.spines.top': False,
            'axes.spines.right': False
        })

    # Create 3x2 subplot layout with tighter spacing
    fig, axes = plt.subplots(3, 2, figsize=(6,9))
    # fig.suptitle(LABELS[LANGUAGE]["symptom_summary"], fontsize=14, fontweight='bold', y=0.96)

    # Flatten axes for easier iteration
    axes_flat = axes.flatten()

    # Create pie chart for each project
    for idx, project in enumerate(projects):
        ax = axes_flat[idx]
        
        # Get symptom data for this project
        project_data = project_symptom_stats[project]
        total_bugs = sum(project_data.values())
        
        # Prepare data for pie chart
        labels = []
        sizes = []
        colors = []
        
        # Sort symptoms by count for this project (descending)
        sorted_symptoms = sorted(project_data.items(), key=lambda x: x[1], reverse=True)
        
        for symptom, count in sorted_symptoms:
            if count > 0:  # Only include symptoms with non-zero counts
                labels.append(symptom)
                sizes.append(count)
                colors.append(symptom_color_map[symptom])
        
        # Create pie chart with academic styling
        wedges, texts, autotexts = ax.pie(sizes, 
                                         labels=None,  # We'll use a legend instead
                                         colors=colors,
                                         autopct=lambda pct: f'{pct:.1f}%' if pct > 3 else '',
                                         startangle=90,
                                         textprops={'fontsize': 7, 'weight': 'bold'},
                                         wedgeprops=dict(width=1.0, edgecolor='white', linewidth=0.8),
                                         pctdistance=0.7)
        
        # Set title with project name and total count (negative padding to bring closer)
        ax.set_title(f'{project}\n(n={total_bugs})', 
                    fontsize=11, 
                    fontweight='bold',
                    pad=-5,  # Negative padding to bring title closer to pie chart
                    y=0.95)   # Position title slightly lower
        
        # Make pie chart circular and remove axes
        ax.set_aspect('equal')
        ax.axis('off')

    # Hide unused subplots if we have fewer than 6 projects
    for idx in range(len(projects), 6):
        axes_flat[idx].axis('off')

    # Create a single legend for all subplots with only the main symptoms
    legend_labels = symptoms
    legend_colors = [symptom_color_map[symptom] for symptom in legend_labels]
    
    # Add compact legend at the very bottom
    fig.legend(legend_labels, 
              loc='lower center', 
              ncol=2,  # Reduced from 4 to 2 columns to make it narrower
              bbox_to_anchor=(0.5, 0.0),  # Position at very bottom
              fontsize=10,
              title=LABELS[LANGUAGE]["symptom"],
              title_fontsize=11,
              frameon=True,
              fancybox=False,
              shadow=False,
              framealpha=0.9,
              edgecolor='black',
              facecolor='white',
              columnspacing=1.0,  # Reduce spacing between columns
              handletextpad=0.5)  # Reduce spacing between legend markers and text

    # Adjust layout to remove all margins and spaces - reserve space for legend
    plt.tight_layout()
    plt.subplots_adjust(top=1.0, bottom=0.15, left=0.0, right=1.0, hspace=0.00, wspace=0.0) 
    
    # Save the plot with no padding
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    print(f"Plot saved as {output_path}")

    # Show the plot
    # plt.show()


def print_summary_stats(project_symptom_stats):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print(LABELS[LANGUAGE]["symptom_summary"])
    print("=" * 70)

    # Sort projects by total bugs
    project_totals = {
        project: sum(stats.values())
        for project, stats in project_symptom_stats.items()
    }
    sorted_projects = sorted(project_totals.items(),
                             key=lambda x: x[1],
                             reverse=True)

    for project, total in sorted_projects:
        stats = project_symptom_stats[project]
        print(f"\n{project}:")
        print(f"  Total Bugs: {total}")
        
        # Sort symptoms by count for this project
        sorted_symptoms = sorted(stats.items(), key=lambda x: x[1], reverse=True)
        for symptom, count in sorted_symptoms:
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"  {symptom}: {count} ({percentage:.1f}%)")


def main():
    # Define file paths
    data_file = Path("../data/final_bug_results_with_root_cause_completed_symptom.json")
    output_file = "symptom_distribution_by_project.png"

    print("Loading and processing bug data...")

    # Load and process data
    project_symptom_stats = load_and_process_data(data_file)

    # Print summary statistics
    print_summary_stats(project_symptom_stats)

    # Create the plot
    print(f"\nCreating symptom distribution plot...")
    create_symptom_distribution_plot(project_symptom_stats, output_file)

    print(f"\nAnalysis complete! Plot saved as {output_file}")


if __name__ == "__main__":
    main()
