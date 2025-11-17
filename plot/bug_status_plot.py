#!/usr/bin/env python3
"""
Script to plot bug status by project.
Creates a bar chart showing three categories for each project:
- Not Confirmed
- Confirmed and Ongoing Fixing  
- Confirmed and Fixed

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

# Static color configuration
COLORS = {
    "Not Confirmed": "#778899",
    "Confirmed and Ongoing Fixing": "#F2CF80",
    "Confirmed and Fixed": "#CD5C5C"
}

# Hatching patterns for better visual distinction
HATCHES = {
    # "Not Confirmed": "/",
    # "Confirmed and Ongoing Fixing": "//",
    # "Confirmed and Fixed": "///"
    "Not Confirmed": "",
    "Confirmed and Ongoing Fixing": "",
    "Confirmed and Fixed": ""
}

# Language configuration
LANGUAGE = "zh"  # Change to "zh" for Chinese version (requires Chinese fonts)

# Text labels based on language
LABELS = {
    "en": {
        "Not Confirmed": "Not Confirmed",
        "Confirmed and Ongoing Fixing": "Confirmed and Ongoing Fixing",
        "Confirmed and Fixed": "Confirmed and Fixed",
        "percentage": "Percentage (%)",
        "bug_status": "Bug Status",
        "bug_status_summary": "BUG STATUS SUMMARY BY PROJECT"
    },
    "zh": {
        "Not Confirmed": "未确认",
        "Confirmed and Ongoing Fixing": "已确认且正在修复",
        "Confirmed and Fixed": "已确认且已修复",
        "percentage": "百分比 (%)",
        "bug_status": "Bug状态",
        "bug_status_summary": "按项目的Bug状态汇总"
    }
}

# plt.rcParams.update({
#     "figure.facecolor": "white",
#     "axes.facecolor": "#FAFAFA",
#     "axes.edgecolor": "#666666",
#     "axes.grid": True,
#     "grid.color": "#E6E6E6",
#     "grid.linewidth": 0.8,
#     "grid.alpha": 1.0,         # 网格清晰但不刺眼
#     "axes.titleweight": "medium",
#     "legend.frameon": False,
# })


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


def categorize_bug_status(bug_data):
    """
    Categorize bug status based on the rules:
    1. Confirmed and Fixed: status == "fixed" and pr_status == "Merged"
    2. Not Confirmed: status == "not_confirmed" 
    3. Confirmed and Ongoing Fixing: everything else
    """
    status = bug_data.get("status", "")
    pr_status = bug_data.get("pr_status", "")
    
    if status == "fixed" and pr_status == "Merged":
        return LABELS[LANGUAGE]["Confirmed and Fixed"]
    elif status == "not_confirmed":
        return LABELS[LANGUAGE]["Not Confirmed"]
    else:
        return LABELS[LANGUAGE]["Confirmed and Ongoing Fixing"]


def load_and_process_data(file_path):
    """Load data and process it by project."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Filter out bugs where is_bug_related is False
    bug_related_data = {
        k: v
        for k, v in data.items() if v.get("is_bug_related") == True
        and v.get("symptom") != "Feature Request"
    }

    # Group by project
    project_stats = defaultdict(
        lambda: {
            LABELS[LANGUAGE]["Not Confirmed"]: 0,
            LABELS[LANGUAGE]["Confirmed and Ongoing Fixing"]: 0,
            LABELS[LANGUAGE]["Confirmed and Fixed"]: 0
        })

    for url, bug_data in bug_related_data.items():
        project_name = extract_project_name(url)
        category = categorize_bug_status(bug_data)
        project_stats[project_name][category] += 1

    return project_stats


def create_bar_plot(project_stats, output_path="bug_status_by_project.png"):
    """Create a percentage-based stacked bar chart of bug status by project."""
    # Define the fixed order of projects
    project_order = [
        "llama.cpp", "vllm", "DeepSpeed", "mlc-llm", "TensorRT-LLM", "TGI"
    ]

    # Filter projects to only include those in our defined order
    projects = [
        project for project in project_order
        if project in project_stats.keys()
    ]

    # Prepare data for plotting
    categories = [
        LABELS[LANGUAGE]["Not Confirmed"], 
        LABELS[LANGUAGE]["Confirmed and Ongoing Fixing"], 
        LABELS[LANGUAGE]["Confirmed and Fixed"]
    ]
    # Map language labels back to English keys for COLORS
    english_categories = []
    for category in categories:
        for key, value in LABELS[LANGUAGE].items():
            if value == category:
                english_categories.append(key)
                break
    
    colors = [COLORS[english_category] for english_category in english_categories]  # Use configurable colors

    # Create percentage data arrays for each category
    percentage_arrays = []
    for category in categories:
        percentages = []
        for project in projects:
            total = sum(project_stats[project].values())
            if total > 0:
                percentage = (project_stats[project][category] / total) * 100
            else:
                percentage = 0
            percentages.append(percentage)
        percentage_arrays.append(percentages)

    # Create the plot with academic styling
    if LANGUAGE == "zh":
        # For Chinese, use a simpler approach that should work better
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Noto Sans CJK JP'],
            'font.size': 10,
            'axes.linewidth': 0.8,
            'grid.linewidth': 0.5,
            'lines.linewidth': 1.0,
            'axes.unicode_minus': False
        })
    else:
        # Use serif fonts for English
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
            'font.size': 10,
            'axes.linewidth': 0.8,
            'grid.linewidth': 0.5,
            'lines.linewidth': 1.0
        })

    fig, ax = plt.subplots(figsize=(12, 8))  # 1:1.5 aspect ratio (12:8 = 1.5:1)

        # Create stacked bars with hatching for better visual distinction
    bottom = np.zeros(len(projects))
    for i, (category,
             percentages) in enumerate(zip(categories, percentage_arrays)):
        # Map language label back to English key for HATCHES
        english_key = None
        for key, value in LABELS[LANGUAGE].items():
            if value == category:
                english_key = key
                break
        
        ax.bar(projects,
               percentages,
               bottom=bottom,
               label=category,
               color=colors[i],
               edgecolor='white',
               linewidth=0.5,
               hatch=HATCHES[english_key] if english_key else "",
               alpha=0.8)
        bottom += percentages

        # Customize the plot with academic styling
    # ax.set_xlabel('Projects', fontsize=11)
    ax.set_ylabel(LABELS[LANGUAGE]["percentage"], fontsize=12)
    # ax.set_title('Bug Status Distribution by Project', fontsize=12, pad=15)
    
    # Set y-axis to show percentages from 0 to 100
    ax.set_ylim(0, 105)
    
    # Add subtle grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Rotate x-axis labels for better readability
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add legend with academic styling
    ax.legend(title=LABELS[LANGUAGE]["bug_status"], fontsize=9,
               frameon=True,
               fancybox=False,
               shadow=False,
               framealpha=0.9)

    # Add percentage labels on bars with academic styling
    for i, project in enumerate(projects):
        total = sum(project_stats[project].values())
        if total > 0:
            # Add total count above the bar with academic font
            ax.text(i,
                    101,
                    f'n={total}',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    color='black',
                    style='italic')

            # Add percentage labels inside the bars
            current_bottom = 0
            for j, category in enumerate(categories):
                percentage = percentage_arrays[j][i]
                if percentage > 1:  # Only show label if segment is large enough (increased threshold)
                    # Use contrasting text color based on segment darkness
                    text_color = 'black'  # White text on dark segment
                    ax.text(i,
                            current_bottom + percentage / 2,
                            f'{percentage:.0f}%',
                            ha='center',
                            va='center',
                            fontsize=8,
                            color=text_color)
                current_bottom += percentage

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_path}")

    # Show the plot
    plt.show()


def print_summary_stats(project_stats):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print(LABELS[LANGUAGE]["bug_status_summary"])
    print("=" * 60)

    # Sort projects by total bugs
    project_totals = {
        project: sum(stats.values())
        for project, stats in project_stats.items()
    }
    sorted_projects = sorted(project_totals.items(),
                             key=lambda x: x[1],
                             reverse=True)

    for project, total in sorted_projects:
        stats = project_stats[project]
        print(f"\n{project}:")
        print(f"  Total Bugs: {total}")
        print(f"  {LABELS[LANGUAGE]['Not Confirmed']}: {stats[LABELS[LANGUAGE]['Not Confirmed']]}")
        print(
            f"  {LABELS[LANGUAGE]['Confirmed and Ongoing Fixing']}: {stats[LABELS[LANGUAGE]['Confirmed and Ongoing Fixing']]}"
        )
        print(f"  {LABELS[LANGUAGE]['Confirmed and Fixed']}: {stats[LABELS[LANGUAGE]['Confirmed and Fixed']]}")

        # Calculate percentages
        if total > 0:
            print(
                f"  Fixed Rate: {stats[LABELS[LANGUAGE]['Confirmed and Fixed']]/total*100:.1f}%")


def main():
    # Define file paths
    data_file = Path(
        "../data/final_bug_results_with_root_cause_completed_symptom.json")
    output_file = "bug_status_by_project.png"

    print("Loading and processing bug data...")

    # Load and process data
    project_stats = load_and_process_data(data_file)

    # Print summary statistics
    print_summary_stats(project_stats)

    # Create the plot
    print(f"\nCreating plot...")
    create_bar_plot(project_stats, output_file)

    print(f"\nAnalysis complete! Plot saved as {output_file}")


if __name__ == "__main__":
    main()
