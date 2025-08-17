#!/usr/bin/env python3
"""
Bug Symptom Classifier - Sample fixed bugs and classify symptoms using GPT
"""

import json
import random
import os
import sys
import time
from typing import List, Dict, Any
import logging
from datetime import datetime

# Import Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Error: Google Generative AI library not installed. Please install it with:")
    print("  pip install google-generativeai")
    sys.exit(1)

# Import OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Error: OpenAI library not installed. Please install it with:")
    print("  pip install openai")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PRE_CALCULATED_SYMPTOMS = ["Runtime Crash", "Incorrect/Inaccuracy Output", "Build/Compilation Error", "Model Loading  Error", "Hardware/Backend Compatibility Issue", "Performance/Memory Issue", "Distributed Parallelism/Sharding Bug", "API/argument parsing Error"]

SAMPLE_SIZE = 2000

# System Prompts (Consolidated - no redundant user prompts)
SYMPTOM_CATEGORIZATION_PROMPT = """You are an expert at analyzing software bugs and categorizing their symptoms. 

Analyze the provided bug reports and identify the main types of symptoms. Return ONLY a JSON array of symptom category names (maximum 8 categories).

Focus on the most common and distinct symptom types. Each category should be:
- Clear and specific (e.g., "compilation error", "runtime crash", ...)
- Broad enough to cover similar issues
- Distinct from other categories

Examples of good categories:
- "compilation error"
- "crash"
- ...

Return ONLY the JSON array, no other text.

Bug reports to analyze:
{bug_summary}"""

INDIVIDUAL_BUG_CLASSIFICATION_PROMPT = """You are an expert at classifying software bugs into specific symptom categories.

Given a bug report, classify it into ONE of these symptom categories:
{symptoms}

If the bug doesn't clearly fit into any of these categories, respond with "other".

Return ONLY the category name, nothing else.

Bug to classify:
{content}"""

class BugSymptomClassifier:
    def __init__(self, api_key: str = None, openai_api_key: str = None):
        # Initialize Gemini client for symptom categorization
        if not GEMINI_AVAILABLE:
            raise ImportError("Google Generative AI library not available. Please install it with: pip install google-generativeai")
        
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available. Please install it with: pip install openai")
        
        # Setup Gemini for symptom categorization
        if not api_key:
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("Google API key not provided and GOOGLE_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)
        self.symptom_client = genai.GenerativeModel("gemini-2.5-pro")
        
        # Setup OpenAI for individual classification
        if not openai_api_key:
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if not openai_api_key:
                raise ValueError("OpenAI API key not provided and OPENAI_API_KEY environment variable not set")
        
        openai.api_key = openai_api_key
        
        # Initialize timing tracking
        self.symptom_response_times = []
        self.classification_response_times = []
        self.symptom_context_lengths = []
        self.classification_context_lengths = []
        self.retry_count = 0
        self.successful_retries = 0
        
        logger.info("Initialized clients: gemini-2.5-pro for symptom categorization, gpt-4.1-mini for individual classification")
    
    def load_fixed_bugs(self, json_file: str = "llm_bugs.json") -> List[Dict[str, Any]]:
        logger.info(f"Loading fixed bugs from {json_file}...")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            fixed_bugs = [bug for bug in data['bugs'] if bug.get('status') == 'fixed']
            logger.info(f"Found {len(fixed_bugs)} fixed bugs out of {len(data['bugs'])} total bugs")
            return fixed_bugs
        except Exception as e:
            logger.error(f"Error loading bugs: {e}")
            return []
    
    def sample_bugs(self, bugs: List[Dict[str, Any]], sample_size: int = 100) -> List[Dict[str, Any]]:
        if len(bugs) <= sample_size:
            logger.info(f"Using all {len(bugs)} fixed bugs")
            return bugs
        sampled_bugs = random.sample(bugs, sample_size)
        logger.info(f"Sampled {len(sampled_bugs)} bugs from {len(bugs)} total fixed bugs")
        return sampled_bugs
    
    def prepare_bug_summary(self, bugs: List[Dict[str, Any]]) -> str:
        summary = "Here are the titles and descriptions of fixed bugs from LLM inference engines:\n\n"
        for i, bug in enumerate(bugs, 1):
            title = bug.get('title', 'No title')
            description = bug.get('description', 'No description')
            repo = bug.get('repository', 'Unknown repo')
            if len(description) > 1000:
                description = description[:1000] + "..."
            summary += f"{i}. Repository: {repo}\n   Title: {title}\n   Description: {description}\n\n"
        return summary
    
    def classify_symptoms(self, bug_summary: str) -> List[str]:
        # Check if PRE_CALCULATED_SYMPTOMS is defined and has exactly 8 items
        if len(PRE_CALCULATED_SYMPTOMS) == 8:
            logger.info("Using pre-calculated symptom categories")
            return PRE_CALCULATED_SYMPTOMS
        
        try:
            # Use consolidated prompt with bug summary included
            full_prompt = SYMPTOM_CATEGORIZATION_PROMPT.format(bug_summary=bug_summary)
            
            # Estimate context length (rough approximation: 1 token ≈ 4 characters)
            context_length = len(full_prompt) // 4
            self.symptom_context_lengths.append(context_length)
            
            # Start timing
            start_time = time.time()
            
            response = self.symptom_client.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=500
                )
            )
            
            # End timing and record
            end_time = time.time()
            response_time = end_time - start_time
            self.symptom_response_times.append(response_time)
            logger.info(f"Symptom categorization API call took {response_time:.2f} seconds")
            
            # Check if response was blocked or failed
            if not response.candidates:
                raise Exception("No response candidates returned from Gemini API")
            
            candidate = response.candidates[0]
            if candidate.finish_reason == 2:  # BLOCKED
                raise Exception("Gemini API blocked the response due to content safety filters")
            elif candidate.finish_reason == 3:  # STOPPED
                raise Exception("Gemini API stopped the response")
            elif candidate.finish_reason == 4:  # MAX_TOKENS
                raise Exception("Gemini API hit maximum token limit")
            elif candidate.finish_reason == 5:  # SAFETY
                raise Exception("Gemini API blocked due to safety concerns")
            
            # Check if response has content
            if not candidate.content or not candidate.content.parts:
                raise Exception("Gemini API returned empty response")
            
            result_text = candidate.content.parts[0].text.strip()
            
            # Parse JSON response
            try:
                # Clean up the response - remove markdown code blocks if present
                cleaned_text = result_text.strip()
                if cleaned_text.startswith('```json'):
                    cleaned_text = cleaned_text[7:]  # Remove ```json
                if cleaned_text.endswith('```'):
                    cleaned_text = cleaned_text[:-3]  # Remove ```
                cleaned_text = cleaned_text.strip()
                
                result = json.loads(cleaned_text)
                
                # Handle different response formats
                if isinstance(result, dict):
                    for key in ['symptoms', 'categories', 'types', 'symptom_types']:
                        if key in result and isinstance(result[key], list):
                            return result[key]
                    for value in result.values():
                        if isinstance(value, list):
                            return value
                elif isinstance(result, list):
                    return result
                else:
                    raise ValueError("Unexpected response format")
                    
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse Gemini response as JSON: {result_text}")
                return self._fallback_symptoms()
                
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return self._fallback_symptoms()
    
    def _classify_single_bug(self, title: str, description: str, symptoms: List[str], retry_count: int = 0) -> str:
        """Classify a single bug into one of the symptom categories using LLM"""
        # Prepare the content for classification
        content = f"Title: {title}\n\nDescription: {description}"
        
        # Truncate if too long
        if len(content) > 2000:
            content = content[:2000] + "..."
        
        try:
            # Create appropriate prompt based on retry count
            if retry_count == 0:
                # First attempt - use original prompt
                full_prompt = INDIVIDUAL_BUG_CLASSIFICATION_PROMPT.format(
                    symptoms=symptoms,
                    content=content
                )
                temperature = 0.1
            else:
                # Retry attempt - add retry instruction to original prompt
                retry_instruction = f"\n\nIMPORTANT: Your previous response was '{getattr(self, '_last_response', 'unknown')}' which is not in the valid categories. Please choose from the exact categories listed above."
                full_prompt = INDIVIDUAL_BUG_CLASSIFICATION_PROMPT.format(
                    symptoms=symptoms,
                    content=content
                ) + retry_instruction
                temperature = 0.0  # Even lower temperature for retry
        
            # Estimate context length (rough approximation: 1 token ≈ 4 characters)
            context_length = len(full_prompt) // 4
            self.classification_context_lengths.append(context_length)
            
            # Start timing
            start_time = time.time()
            
            response = openai.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": full_prompt}],
                temperature=temperature,
                max_tokens=50
            )
            
            # End timing and record
            end_time = time.time()
            response_time = end_time - start_time
            self.classification_response_times.append(response_time)
            
            # Check if response was blocked or failed
            if not response.choices:
                raise Exception("No response candidates returned from OpenAI API")
            
            result_text = response.choices[0].message.content.strip().lower()
            
            # Clean up the response
            result_text = result_text.replace('"', '').replace("'", "").strip()
            
            # Store the response for potential retry
            self._last_response = result_text
            
            # Check if the result matches any of our symptoms
            for symptom in symptoms:
                if symptom.lower() in result_text or result_text in symptom.lower():
                    if retry_count > 0:
                        logger.debug(f"Retry successful for '{title[:50]}...': '{getattr(self, '_last_response', 'unknown')}' -> '{result_text}'")
                        self.successful_retries += 1
                    return symptom
            
            # If no match found and we haven't retried yet, try again
            if retry_count < 1:
                self.retry_count += 1
                logger.debug(f"Retrying classification for '{title[:50]}...' - original response: '{result_text}'")
                return self._classify_single_bug(title, description, symptoms, retry_count + 1)
            
            # If still no match after retry, return "other"
            logger.debug(f"No symptom match found for '{title[:50]}...' after retry. Response: '{result_text}'. Returning 'other'")
            return "other"
                
        except Exception as e:
            logger.error(f"Error classifying bug '{title[:50]}...': {e}")
            return "other"
    
    def _fallback_symptoms(self) -> List[str]:
        logger.warning("Using fallback symptom categories")
        return [
            "compilation error",
            "runtime crash",
            "memory leak",
            "performance issue",
            "incorrect output",
            "API error",
            "segmentation fault",
            "deadlock"
        ]
    
    def analyze_bug_distribution(self, bugs: List[Dict[str, Any]], symptoms: List[str]) -> Dict[str, Any]:
        """Analyze bug distribution by classifying each bug into symptom categories using LLM"""
        repo_stats = {}
        type_stats = {'issue': 0, 'pr': 0}
        symptom_stats = {symptom: 0 for symptom in symptoms}
        symptom_stats['other'] = 0  # For bugs that don't fit clear categories
        
        # Add detailed classification for each bug
        bug_classifications = []
        
        logger.info(f"🔍 Classifying {len(bugs)} bugs into {len(symptoms)} symptom categories...")
        
        for i, bug in enumerate(bugs, 1):
            repo = bug.get('repository', 'Unknown')
            bug_type = bug.get('type', 'unknown')
            title = bug.get('title', '')
            description = bug.get('description', '')
            
            # Update repository and type stats
            if repo not in repo_stats:
                repo_stats[repo] = {'total': 0, 'issues': 0, 'prs': 0}
            repo_stats[repo]['total'] += 1
            if bug_type == 'issue':
                repo_stats[repo]['issues'] += 1
            else:
                repo_stats[repo]['prs'] += 1
            
            if bug_type in type_stats:
                type_stats[bug_type] += 1
            
            # Classify this specific bug using LLM
            symptom_category = self._classify_single_bug(title, description, symptoms)
            
            # Update symptom statistics
            if symptom_category in symptom_stats:
                symptom_stats[symptom_category] += 1
            else:
                symptom_stats['other'] += 1
            
            # Store classification details
            bug_classifications.append({
                'repository': repo,
                'type': bug_type,
                'title': title,
                'url': bug.get('url', ''),
                'number': bug.get('number', 0),
                'classified_symptom': symptom_category,
                'confidence': 'high' if symptom_category in symptoms else 'low'
            })
            
            # Progress logging
            if i % 50 == 0:
                logger.info(f"📊 Classified {i}/{len(bugs)} bugs...")
        
        logger.info(f"✅ Completed classification of {len(bugs)} bugs")
        
        return {
            'total_bugs': len(bugs),
            'repositories': repo_stats,
            'types': type_stats,
            'symptoms': symptoms,
            'symptom_distribution': symptom_stats,
            'bug_classifications': bug_classifications,
            'classification_summary': {
                'total_classified': len(bugs),
                'symptoms_used': len([s for s in symptom_stats.values() if s > 0]),
                'other_count': symptom_stats['other'],
                'other_percentage': (symptom_stats['other'] / len(bugs)) * 100 if len(bugs) > 0 else 0
            }
        }
    
    def print_average_response_times_and_context_len(self):
        """Print average response times and context lengths for LLM API calls"""
        print("\n⏱️  LLM Response Time and Context Length Statistics:")
        print("=" * 60)
        
        if self.symptom_response_times:
            avg_symptom_time = sum(self.symptom_response_times) / len(self.symptom_response_times)
            min_symptom_time = min(self.symptom_response_times)
            max_symptom_time = max(self.symptom_response_times)
            avg_symptom_context = sum(self.symptom_context_lengths) / len(self.symptom_context_lengths)
            min_symptom_context = min(self.symptom_context_lengths)
            max_symptom_context = max(self.symptom_context_lengths)
            print(f"🔍 Symptom Categorization (Gemini 2.5-pro):")
            print(f"   Average time: {avg_symptom_time:.2f} seconds")
            print(f"   Min time: {min_symptom_time:.2f} seconds")
            print(f"   Max time: {max_symptom_time:.2f} seconds")
            print(f"   Average context length: {avg_symptom_context:.0f} tokens")
            print(f"   Min context length: {min_symptom_context:.0f} tokens")
            print(f"   Max context length: {max_symptom_context:.0f} tokens")
            print(f"   Total calls: {len(self.symptom_response_times)}")
        else:
            print("🔍 Symptom Categorization: No calls made")
        
        if self.classification_response_times:
            avg_classification_time = sum(self.classification_response_times) / len(self.classification_response_times)
            min_classification_time = min(self.classification_response_times)
            max_classification_time = max(self.classification_response_times)
            avg_classification_context = sum(self.classification_context_lengths) / len(self.classification_context_lengths)
            min_classification_context = min(self.classification_context_lengths)
            max_classification_context = max(self.classification_context_lengths)
            print(f"🏷️  Individual Classification (gpt-4.1-mini):")
            print(f"   Average time: {avg_classification_time:.2f} seconds")
            print(f"   Min time: {min_classification_time:.2f} seconds")
            print(f"   Max time: {max_classification_time:.2f} seconds")
            print(f"   Average context length: {avg_classification_context:.0f} tokens")
            print(f"   Min context length: {min_classification_context:.0f} tokens")
            print(f"   Max context length: {max_classification_context:.0f} tokens")
            print(f"   Total calls: {len(self.classification_response_times)}")
        else:
            print("🏷️  Individual Classification: No calls made")
        
        # Calculate overall statistics
        all_times = self.symptom_response_times + self.classification_response_times
        all_contexts = self.symptom_context_lengths + self.classification_context_lengths
        if all_times:
            total_avg_time = sum(all_times) / len(all_times)
            total_time = sum(all_times)
            total_avg_context = sum(all_contexts) / len(all_contexts)
            total_context = sum(all_contexts)
            print(f"\n📊 Overall Statistics:")
            print(f"   Total API calls: {len(all_times)}")
            print(f"   Overall average time: {total_avg_time:.2f} seconds")
            print(f"   Overall average context length: {total_avg_context:.0f} tokens")
            print(f"   Total time spent: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
            print(f"   Total tokens used: {total_context:.0f} tokens")
        
        # Print retry statistics
        if self.retry_count > 0:
            success_rate = (self.successful_retries / self.retry_count) * 100 if self.retry_count > 0 else 0
            print(f"\n🔄 Retry Statistics:")
            print(f"   Total retries: {self.retry_count}")
            print(f"   Successful retries: {self.successful_retries}")
            print(f"   Retry success rate: {success_rate:.1f}%")
        else:
            print(f"\n🔄 Retry Statistics: No retries needed")

    def save_results(self, results: Dict[str, Any], filename: str = None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bug_symptoms_{timestamp}.json"
        
        output_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'analysis_method': 'Gemini symptom classification',
                'provider': 'gemini',
                'sample_size': results['total_bugs']
            },
            'symptoms': results['symptoms'],
            'statistics': {
                'total_bugs_analyzed': results['total_bugs'],
                'repository_distribution': results['repositories'],
                'type_distribution': results['types'],
                'symptom_distribution': results['symptom_distribution'],
                'classification_summary': results['classification_summary']
            },
            'detailed_classifications': results['bug_classifications']
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {filename}")
        return filename

def main():
    print("🐛 Bug Symptom Classifier")
    print("=" * 50)
    
    # Parse command line arguments
    sample_size = SAMPLE_SIZE
    input_file = "llm_bugs.json"
    
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--sample-size" and i + 1 < len(sys.argv):
            try:
                sample_size = int(sys.argv[i + 1])
                if sample_size <= 0:
                    raise ValueError("Sample size must be positive")
                i += 2
            except ValueError:
                print(f"Error: Invalid sample size '{sys.argv[i + 1]}'. Using default of {sample_size}.")
                i += 2
        elif arg == "--input-file" and i + 1 < len(sys.argv):
            input_file = sys.argv[i + 1]
            i += 2
        elif arg == "--help":
            print("Usage: python classifier.py [options]")
            print("Options:")
            print(f"  --sample-size N     Number of bugs to sample (default: {sample_size})")
            print("  --input-file FILE   Input JSON file (default: llm_bugs.json)")
            print("  --help              Show this help message")
            print()
            print("Environment Variables:")
            print("  GOOGLE_API_KEY      Google API key (required)")
            print("  OPENAI_API_KEY      OpenAI API key (required)")
            return
        else:
            # Legacy support for positional arguments
            if i == 1:
                try:
                    sample_size = int(arg)
                    if sample_size <= 0:
                        raise ValueError("Sample size must be positive")
                except ValueError:
                    print(f"Error: Invalid sample size '{arg}'. Using default of {sample_size}.")
            elif i == 2:
                input_file = arg
            i += 1
    
    print(f"📊 Sample size: {sample_size}")
    print(f"📁 Input file: {input_file}")
    print()
    
    try:
        # Get Google API key
        api_key = os.getenv('GOOGLE_API_KEY')
        openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            print("Error: GOOGLE_API_KEY environment variable not set")
            print("Please set your Google API key: export GOOGLE_API_KEY='your_key_here'")
            print("Get your API key from: https://makersuite.google.com/app/apikey")
            return
        
        if not openai_api_key:
            print("Error: OPENAI_API_KEY environment variable not set")
            print("Please set your OpenAI API key: export OPENAI_API_KEY='your_key_here'")
            print("Get your API key from: https://platform.openai.com/api-keys")
            return
        
        classifier = BugSymptomClassifier(api_key=api_key, openai_api_key=openai_api_key)
        print("🤖 Using Gemini 2.5-pro for symptom categorization and gpt-4.1-mini for individual classification")
        
        fixed_bugs = classifier.load_fixed_bugs(input_file)
        if not fixed_bugs:
            print("❌ No fixed bugs found. Exiting.")
            return
        
        sampled_bugs = classifier.sample_bugs(fixed_bugs, sample_size)
        print("🤖 Analyzing bug symptoms with Gemini...")
        bug_summary = classifier.prepare_bug_summary(sampled_bugs)
        symptoms = classifier.classify_symptoms(bug_summary)
        results = classifier.analyze_bug_distribution(sampled_bugs, symptoms)
        output_file = classifier.save_results(results)
        
        print("\n📋 Classification Results:")
        print("=" * 50)
        print(f"🔍 Identified {len(symptoms)} symptom categories:")
        for i, symptom in enumerate(symptoms, 1):
            print(f"   {i}. {symptom}")
        
        print(f"\n📊 Basic Statistics:")
        print(f"   Total bugs analyzed: {results['total_bugs']}")
        print(f"   Issues: {results['types']['issue']}")
        print(f"   Pull Requests: {results['types']['pr']}")
        
        print(f"\n🏢 Repository Distribution:")
        for repo, stats in results['repositories'].items():
            print(f"   {repo}: {stats['total']} bugs ({stats['issues']} issues, {stats['prs']} PRs)")
        
        print(f"\n🎯 Symptom Distribution (LLM Classification):")
        symptom_dist = results['symptom_distribution']
        total_classified = sum(symptom_dist.values())
        
        # Sort symptoms by count (descending)
        sorted_symptoms = sorted(symptom_dist.items(), key=lambda x: x[1], reverse=True)
        
        for symptom, count in sorted_symptoms:
            if count > 0:
                percentage = (count / total_classified) * 100
                print(f"   {symptom}: {count} bugs ({percentage:.1f}%)")
        
        print(f"\n📈 Classification Summary:")
        summary = results['classification_summary']
        print(f"   Total classified: {summary['total_classified']}")
        print(f"   Symptoms used: {summary['symptoms_used']}")
        print(f"   Other category: {summary['other_count']} bugs ({summary['other_percentage']:.1f}%)")
        
        print(f"\n💾 Results saved to: {output_file}")
        
        # Print average response times
        classifier.print_average_response_times_and_context_len()
        
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user.")
        return
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        print(f"❌ Error: {e}")
        return

if __name__ == "__main__":
    main()
