#!/usr/bin/env python3
"""
Bug Symptom Classifier - Classify fixed and merged bugs into predefined symptom categories using GPT
"""

import json
import os
import sys
import time
from typing import List, Dict, Any
import logging
from datetime import datetime

# Import OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PRE_CALCULATED_SYMPTOMS = [
    "Security vulnerability", "Runtime Crash", "Incorrect/Inaccuracy Output",
    "Build/Compilation Error", "Model Loading Error",
    "Hardware/Backend Compatibility Issue", "Performance/Memory Issue",
    "Distributed Parallelism/Sharding Bug", "API/argument parsing Error",
    "Feature Request"
]

# System prompt for bug classification
SYSTEM_PROMPT = """You are an expert software bug analyst specializing in categorizing software defects and issues. Your task is to classify bugs into predefined symptom categories based on their technical characteristics.

You must respond with EXACTLY one of the provided category names, or "other: [suggestion]" if none fit.

Guidelines:
- Focus on the technical symptom, not the root cause
- Copy category names EXACTLY as provided (including spacing/punctuation)
- Only use "other:" if truly none of the categories match
- Be consistent in your classifications"""

# User prompt template for individual bugs
USER_PROMPT_TEMPLATE = """Classify this bug into the BEST category from this list:

{symptoms}

Bug details:
Title: {title}
Description: {description}  
ReasoningForThisBug: {reasoning}

Response format: Either copy one category name exactly, or "other: [your suggestion]"

Category:"""


class BugSymptomClassifier:
        
    def __init__(self, openai_api_key: str = None):
        # Only need OpenAI for classification since we use predefined symptoms
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI library not available. Please install it with: pip install openai"
            )
        
        # Setup OpenAI for individual classification
        if not openai_api_key:
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if not openai_api_key:
                raise ValueError(
                    "OpenAI API key not provided and OPENAI_API_KEY environment variable not set"
                )
        
        openai.api_key = openai_api_key
        
        # Initialize timing tracking
        self.classification_response_times = []
        self.classification_context_lengths = []
        self.retry_count = 0
        self.successful_retries = 0
        
        logger.info(
            "Initialized client: gpt-5-nano for individual classification")

    def load_all_bugs(
        self,
        json_file: str = "data/final_bug_results_with_root_cause_completed.json"
    ) -> List[Dict[str, Any]]:
        logger.info(f"Loading all bugs from {json_file}...")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            all_bugs = []
            for url, bug_data in data.items():
                # Add the URL to the bug data for reference
                bug_data['url'] = url
                all_bugs.append(bug_data)

            logger.info(f"Found {len(all_bugs)} total bugs")
            return all_bugs
        except Exception as e:
            logger.error(f"Error loading bugs: {e}")
            return []
    
    def load_cache(self, cache_file: str) -> Dict[str, Dict[str, Any]]:
        """Load existing cache file if it exists"""
        try:
            if os.path.exists(cache_file):
                logger.info(f"Loading cache from {cache_file}...")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                logger.info(f"Loaded {len(cache_data)} cached bug classifications")
                return cache_data
            else:
                logger.info(f"Cache file {cache_file} does not exist, starting fresh")
                return {}
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return {}

    # Remove sampling since we want to process all bugs
    # def sample_bugs method is no longer needed

    # Remove prepare_bug_summary since we don't need symptom categorization anymore

    def get_symptoms(self) -> List[str]:
        """Return the predefined symptom categories"""
        logger.info("Using predefined symptom categories")
        return PRE_CALCULATED_SYMPTOMS

    def _classify_single_bug(self,
                             title: str,
                             description: str,
                             reasoning: str,
                             symptoms: List[str],
                             retry_count: int = 0) -> str:
        """Classify a single bug into one of the symptom categories using LLM"""
        
        # Use full content without truncation
        
        try:
            # Create appropriate prompt based on retry count
            if retry_count == 0:
                # First attempt - use system and user prompts
                user_prompt = USER_PROMPT_TEMPLATE.format(
                    symptoms='\n'.join([
                        f"{i+1}. {symptom}"
                        for i, symptom in enumerate(symptoms)
                    ]),
                    title=title,
                    description=description,
                    reasoning=reasoning)
                messages = [{
                    "role": "system",
                    "content": SYSTEM_PROMPT
                }, {
                    "role": "user",
                    "content": user_prompt
                }]
                temperature = 0.1
            else:
                # Retry attempt - add retry instruction
                retry_instruction = f"\n\nIMPORTANT: Your previous response '{getattr(self, '_last_response', 'unknown')}' was not recognized. You must respond with EXACTLY one of the category names listed above (copy exactly), or 'other: <your_suggestion>' if none fit."
                user_prompt = USER_PROMPT_TEMPLATE.format(
                    symptoms='\n'.join([
                        f"{i+1}. {symptom}"
                        for i, symptom in enumerate(symptoms)
                    ]),
                    title=title,
                    description=description,
                    reasoning=reasoning) + retry_instruction
                messages = [{
                    "role": "system",
                    "content": SYSTEM_PROMPT
                }, {
                    "role": "user",
                    "content": user_prompt
                }]
                temperature = 0.0  # Even lower temperature for retry
        
            # Estimate context length (rough approximation: 1 token ≈ 4 characters)
            total_content = SYSTEM_PROMPT + user_prompt
            context_length = len(total_content) // 4
            self.classification_context_lengths.append(context_length)
            
            # Start timing
            start_time = time.time()
            
            response = openai.chat.completions.create(
                model="gpt-5-nano",  # reasoning model
                messages=messages)
            
            # End timing and record
            end_time = time.time()
            response_time = end_time - start_time
            self.classification_response_times.append(response_time)
            
            # Check if response was blocked or failed
            if not response.choices:
                raise Exception(
                    "No response candidates returned from OpenAI API")
            
            result_text = response.choices[0].message.content.strip()
            
            # Clean up the response
            result_text = result_text.replace('"', '').replace("'", "").strip()
            
            # Store the response for potential retry
            self._last_response = result_text
            
            # Check if result starts with "other:"
            if result_text.lower().startswith("other:"):
                return result_text  # Return as-is for "other: <symptom>" format

            # Check for exact match first (case-insensitive)
            for symptom in symptoms:
                if symptom.lower() == result_text.lower():
                    if retry_count > 0:
                        logger.debug(
                            f"Retry successful for '{title[:50]}...': '{getattr(self, '_last_response', 'unknown')}' -> '{result_text}'"
                        )
                        self.successful_retries += 1
                    return symptom

            # Check for partial matches (more lenient)
            for symptom in symptoms:
                if (symptom.lower() in result_text.lower()
                        or result_text.lower() in symptom.lower()):
                    if retry_count > 0:
                        logger.debug(
                            f"Retry successful for '{title[:50]}...': '{getattr(self, '_last_response', 'unknown')}' -> '{result_text}'"
                        )
                        self.successful_retries += 1
                    return symptom
            
            # If no match found and we haven't retried yet, try again
            if retry_count < 1:
                self.retry_count += 1
                logger.debug(
                    f"Retrying classification for '{title[:50]}...' - original response: '{result_text}'"
                )
                return self._classify_single_bug(title, description, reasoning,
                                                 symptoms, retry_count + 1)

            # If still no match after retry, return "other: <original_response>"
            logger.debug(
                f"No symptom match found for '{title[:50]}...' after retry. Response: '{result_text}'. Returning 'other: {result_text}'"
            )
            return f"other: {result_text}"
                
        except Exception as e:
            logger.error(f"Error classifying bug '{title[:50]}...': {e}")
            return "other: classification_error"

    def _print_symptom_distribution(self,
                                    symptom_stats: Dict[str, int],
                                    other_stats: Dict[str, int],
                                    total_bugs: int,
                                    show_all_other: bool = False):
        """Print symptom distribution statistics"""
        # Print predefined symptom distribution
        total_predefined = sum(symptom_stats.values())
        if total_predefined > 0:
            print(f"🎯 Predefined Symptoms ({total_predefined} bugs):")
            sorted_symptoms = sorted(symptom_stats.items(),
                                     key=lambda x: x[1],
                                     reverse=True)
            for symptom, count in sorted_symptoms:
                if count > 0:
                    percentage = (count / total_bugs) * 100
                    print(f"   {symptom}: {count} ({percentage:.1f}%)")

        # Print other categories
        total_other = sum(other_stats.values())
        if total_other > 0:
            print(f"🔍 Other Categories ({total_other} bugs):")
            sorted_other = sorted(other_stats.items(),
                                  key=lambda x: x[1],
                                  reverse=True)

            if show_all_other:
                # Show all other categories (for final statistics)
                for category, count in sorted_other:
                    percentage = (count / total_bugs) * 100
                    print(f"   {category}: {count} ({percentage:.1f}%)")
            else:
                # Show top 5 other categories (for periodic updates)
                for category, count in sorted_other[:5]:
                    percentage = (count / total_bugs) * 100
                    print(f"   {category}: {count} ({percentage:.1f}%)")
                if len(other_stats) > 5:
                    remaining = len(other_stats) - 5
                    print(f"   ... and {remaining} more categories")

    def _print_timing_statistics(self,
                                 is_final: bool = False,
                                 current: int = 0,
                                 total: int = 0):
        """Print timing and performance statistics"""
        if not self.classification_response_times:
            print("🏷️  Classification: No calls made")
            return

        avg_time = sum(self.classification_response_times) / len(
            self.classification_response_times)
        min_time = min(self.classification_response_times)
        max_time = max(self.classification_response_times)
        total_time = sum(self.classification_response_times)

        avg_context = sum(self.classification_context_lengths) / len(
            self.classification_context_lengths)
        min_context = min(self.classification_context_lengths)
        max_context = max(self.classification_context_lengths)
        total_context = sum(self.classification_context_lengths)

        if is_final:
            print(f"🏷️  Classification Statistics (gpt-5-nano):")
            print(f"   Average time: {avg_time:.2f} seconds")
            print(f"   Min time: {min_time:.2f} seconds")
            print(f"   Max time: {max_time:.2f} seconds")
            print(f"   Average context length: {avg_context:.0f} tokens")
            print(f"   Min context length: {min_context:.0f} tokens")
            print(f"   Max context length: {max_context:.0f} tokens")
            print(f"   Total calls: {len(self.classification_response_times)}")
            print(
                f"   Total time spent: {total_time:.2f} seconds ({total_time/60:.1f} minutes)"
            )
            print(f"   Total tokens used: {total_context:.0f} tokens")
        else:
            estimated_remaining = (
                total -
                current) * avg_time if current > 0 and total > current else 0
            print(
                f"⏱️  Timing: Avg {avg_time:.2f}s/bug, Total {total_time:.1f}s, Est. remaining {estimated_remaining/60:.1f}min"
            )

    def _print_periodic_statistics(self, current: int, total: int,
                                   symptom_stats: Dict[str, int],
                                   other_stats: Dict[str, int]):
        """Print periodic statistics during classification"""
        print(f"\n📊 Progress Statistics ({current}/{total} bugs classified):")
        print("=" * 50)

        self._print_symptom_distribution(symptom_stats,
                                         other_stats,
                                         current,
                                         show_all_other=False)
        self._print_timing_statistics(is_final=False,
                                      current=current,
                                      total=total)

        print("=" * 50)

    # Remove _fallback_symptoms since we use predefined symptoms

    def classify_all_bugs_with_cache(self, bugs: List[Dict[str, Any]],
                                    symptoms: List[str], 
                                    cache_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Classify only fixed/merged bugs using cache when available, preserve all others"""
        logger.info(f"🔍 Processing {len(bugs)} total bugs...")

        # Statistics tracking
        symptom_stats = {symptom: 0 for symptom in symptoms}
        other_stats = {}  # Track different "other" categories
        
        # Counters for cache usage and bug types
        cache_hits = 0
        new_classifications = 0
        fixed_merged_count = 0
        preserved_count = 0

        # Process each bug
        classified_bugs = []
        
        for i, bug in enumerate(bugs, 1):
            url = bug.get('url', '')
            
            # Check if this bug should be classified (fixed and merged)
            should_classify = (bug.get('status') == 'fixed' and
                             bug.get('pr_status') == 'Merged')
            
            if should_classify:
                fixed_merged_count += 1
                
                # Check if this bug is already in cache and has a symptom
                if url in cache_data and 'symptom' in cache_data[url]:
                    # Use cached classification
                    classified_bug = bug.copy()
                    classified_bug['symptom'] = cache_data[url]['symptom']
                    cache_hits += 1
                    symptom_category = cache_data[url]['symptom']
                else:
                    # Need to classify this bug
                    title = bug.get('title', '')
                    description = bug.get('description', '')
                    reasoning = bug.get('reasoning_for_bug_related', '')

                    # Classify this specific bug using LLM
                    symptom_category = self._classify_single_bug(
                        title, description, reasoning, symptoms)

                    # Create a copy of the bug data and add symptom classification
                    classified_bug = bug.copy()
                    classified_bug['symptom'] = symptom_category
                    new_classifications += 1

                # Update statistics for classified bugs only
                if symptom_category in symptoms:
                    symptom_stats[symptom_category] += 1
                elif symptom_category.startswith('other:'):
                    if symptom_category not in other_stats:
                        other_stats[symptom_category] = 0
                    other_stats[symptom_category] += 1
                else:
                    # Fallback for unexpected responses
                    if 'other: unexpected_response' not in other_stats:
                        other_stats['other: unexpected_response'] = 0
                    other_stats['other: unexpected_response'] += 1
            else:
                # Bug is not fixed/merged - preserve as-is without classification
                classified_bug = bug.copy()
                preserved_count += 1

            classified_bugs.append(classified_bug)

            # Progress logging with periodic statistics
            if i % 50 == 0:
                logger.info(f"📊 Processed {i}/{len(bugs)} bugs (Fixed/Merged: {fixed_merged_count}, Preserved: {preserved_count}, Cache: {cache_hits}, New: {new_classifications})...")
                if fixed_merged_count > 0:  # Only show stats if we have classified bugs
                    self._print_periodic_statistics(fixed_merged_count, fixed_merged_count, symptom_stats,
                                                    other_stats)

        logger.info(f"✅ Completed processing {len(bugs)} total bugs")
        logger.info(f"📈 Classification Summary: {fixed_merged_count} fixed/merged bugs classified, {preserved_count} other bugs preserved")
        logger.info(f"📈 Cache Performance: {cache_hits} hits, {new_classifications} new classifications")

        # Combine symptom stats and other stats for reporting
        all_symptom_stats = symptom_stats.copy()
        all_symptom_stats.update(other_stats)

        return {
            'classified_bugs': classified_bugs,
            'statistics': {
                'total_bugs': len(bugs),
                'fixed_merged_bugs': fixed_merged_count,
                'preserved_bugs': preserved_count,
                'predefined_symptoms': symptom_stats,
                'other_categories': other_stats,
                'all_symptom_distribution': all_symptom_stats,
                'cache_performance': {
                    'cache_hits': cache_hits,
                    'new_classifications': new_classifications,
                    'cache_hit_rate': (cache_hits / fixed_merged_count) * 100 if fixed_merged_count > 0 else 0
                }
            }
        }

    def print_classification_statistics(self,
                                        symptom_stats: Dict[str, int] = None,
                                        other_stats: Dict[str, int] = None,
                                        total_bugs: int = 0):
        """Print final classification statistics and response times"""
        print("\n⏱️  Final Classification Statistics:")
        print("=" * 60)
        
        # Print symptom distribution if provided
        if symptom_stats is not None and other_stats is not None and total_bugs > 0:
            self._print_symptom_distribution(symptom_stats,
                                             other_stats,
                                             total_bugs,
                                             show_all_other=True)
            print()

        # Print timing statistics
        self._print_timing_statistics(is_final=True)
        
        # Print retry statistics
        if self.retry_count > 0:
            success_rate = (self.successful_retries / self.retry_count
                            ) * 100 if self.retry_count > 0 else 0
            print(f"\n🔄 Retry Statistics:")
            print(f"   Total retries: {self.retry_count}")
            print(f"   Successful retries: {self.successful_retries}")
            print(f"   Retry success rate: {success_rate:.1f}%")
        else:
            print(f"\n🔄 Retry Statistics: No retries needed")

    def save_classified_bugs(self,
                             classified_bugs: List[Dict[str, Any]],
                             filename: str = None):
        """Save the classified bugs with symptom attributes back to JSON"""
        # Convert list back to dictionary format like the original
        output_data = {}
        for bug in classified_bugs:
            url = bug.get('url', '')
            if url:
                # Remove the url from the bug data since it's used as key
                bug_data = bug.copy()
                del bug_data['url']
                output_data[url] = bug_data
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Classified bugs saved to {filename}")
        return filename


def main():
    print("🐛 Bug Symptom Classifier")
    print("=" * 50)
    
    # Parse command line arguments
    input_file = "data/final_bug_results_with_root_cause_completed.json"
    output_file = "data/final_bug_results_with_root_cause_completed_symptom.json"
    
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--input-file" and i + 1 < len(sys.argv):
            input_file = sys.argv[i + 1]
            i += 2
        elif arg == "--output-file" and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]
            i += 2
        elif arg == "--help":
            print("Usage: python classifier.py [options]")
            print("Options:")
            print(
                f"  --input-file FILE    Input JSON file (default: {input_file})"
            )
            print(
                "  --output-file FILE   Output JSON file (default: auto-generated)"
            )
            print("  --help               Show this help message")
            print()
            print("Environment Variables:")
            print("  OPENAI_API_KEY      OpenAI API key (required)")
            return
        else:
            # Legacy support for positional arguments
            if i == 1:
                input_file = arg
            elif i == 2:
                output_file = arg
            i += 1
    
    print(f"📁 Input file: {input_file}")
    if output_file:
        print(f"📁 Output file: {output_file}")
    print()
    
    try:
        # Get OpenAI API key
        openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not openai_api_key:
            print("Error: OPENAI_API_KEY environment variable not set")
            print(
                "Please set your OpenAI API key: export OPENAI_API_KEY='your_key_here'"
            )
            print(
                "Get your API key from: https://platform.openai.com/api-keys")
            return
        
        classifier = BugSymptomClassifier(openai_api_key=openai_api_key)
        print("🤖 Using gpt-5-nano for individual bug classification")

        # Load all bugs from input file
        all_bugs = classifier.load_all_bugs(input_file)
        if not all_bugs:
            print("❌ No bugs found. Exiting.")
            return
        
        print(f"📊 Found {len(all_bugs)} total bugs to process")

        # Load cache if it exists
        cache_data = classifier.load_cache(output_file)

        # Get predefined symptoms
        symptoms = classifier.get_symptoms()
        print(f"🎯 Using {len(symptoms)} predefined symptom categories:")
        for i, symptom in enumerate(symptoms, 1):
            print(f"   {i}. {symptom}")
        print()

        # Process all bugs with caching
        print("🤖 Starting classification with caching...")
        results = classifier.classify_all_bugs_with_cache(all_bugs, symptoms, cache_data)

        # Save results
        saved_file = classifier.save_classified_bugs(
            results['classified_bugs'], output_file)

        # Print results summary
        print("\n📋 Classification Results:")
        print("=" * 50)

        stats = results['statistics']
        print(f"📊 Total bugs processed: {stats['total_bugs']}")
        print(f"📊 Fixed/Merged bugs classified: {stats['fixed_merged_bugs']}")
        print(f"📊 Other bugs preserved: {stats['preserved_bugs']}")

        # Only show symptom distribution if we have classified bugs
        if stats['fixed_merged_bugs'] > 0:
            print(f"\n🎯 Predefined Symptom Distribution (Fixed/Merged bugs only):")
            predefined_stats = stats['predefined_symptoms']
            total_predefined = sum(predefined_stats.values())

            # Sort symptoms by count (descending)
            sorted_predefined = sorted(predefined_stats.items(),
                                        key=lambda x: x[1],
                                        reverse=True)

            for symptom, count in sorted_predefined:
                if count > 0:
                    percentage = (count / stats['fixed_merged_bugs']) * 100
                    print(f"   {symptom}: {count} bugs ({percentage:.1f}%)")

            print(f"\n🔍 Other Categories:")
            other_stats = stats['other_categories']
            total_other = sum(other_stats.values())

            if other_stats:
                # Sort other categories by count (descending)
                sorted_other = sorted(other_stats.items(),
                                       key=lambda x: x[1],
                                       reverse=True)

                for category, count in sorted_other:
                    percentage = (count / stats['fixed_merged_bugs']) * 100
                    print(f"   {category}: {count} bugs ({percentage:.1f}%)")
            else:
                print("   None - all fixed/merged bugs classified into predefined categories!")

            print(f"\n📈 Classification Summary:")
            print(
                f"   Predefined categories: {total_predefined} bugs ({(total_predefined/stats['fixed_merged_bugs'])*100:.1f}%)"
            )
            print(
                f"   Other categories: {total_other} bugs ({(total_other/stats['fixed_merged_bugs'])*100:.1f}%)"
            )
        else:
            print(f"\n📊 No fixed/merged bugs found to classify")

        # Print cache performance
        cache_perf = stats['cache_performance']
        print(f"\n🚀 Cache Performance:")
        print(f"   Cache hits: {cache_perf['cache_hits']} bugs")
        print(f"   New classifications: {cache_perf['new_classifications']} bugs")
        print(f"   Cache hit rate: {cache_perf['cache_hit_rate']:.1f}%")

        print(f"\n💾 All bugs saved to: {saved_file}")

        # Print final classification statistics
        classifier.print_classification_statistics(
            symptom_stats=stats['predefined_symptoms'],
            other_stats=stats['other_categories'],
            total_bugs=stats['total_bugs'])
        
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user.")
        return
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        print(f"❌ Error: {e}")
        return


if __name__ == "__main__":
    main()
