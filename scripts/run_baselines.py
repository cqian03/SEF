#!/usr/bin/env python3
"""
Run baseline experiments for SEF paper across all domains.

Supports iterative evaluation with majority voting.

Usage:
    # Run all baselines with 5 iterations (majority voting)
    python scripts/run_baselines.py --model qwen_2_5_14b --domain all --iterations 5
    
    # Run specific method
    python scripts/run_baselines.py --model qwen_2_5_14b --method standard_cot --iterations 5
    
    # Run single iteration (original behavior)
    python scripts/run_baselines.py --model qwen_2_5_14b --domain legal
"""

import os
import sys
import argparse
import json
import logging
import re
from pathlib import Path
from datetime import datetime
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import List, Dict, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from src.data_loader import MultiDomainLoader
from src.llm_clients import get_client
from src.baselines import get_baseline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Models used in the paper (Section 4.1)
MODELS = [
    'deepseek_r1_14b',
    'gemma_3_12b',
    'ministral_14b',
    'qwen_2_5_14b',
]

# Baseline methods used in the paper (Section 4.1)
METHODS = [
    'direct',
    'standard_cot',
    'tree_of_thought',
    'chain_of_verification',
    'vanilla_rag',
    'self_rag',
]

DOMAINS = ['legal', 'medical', 'financial']


def extract_answer_from_response(response: str) -> str:
    """
    Extract Yes/No answer from response.
    
    Handles various formats including:
    - "Answer: Yes"
    - "**Answer:**\n1. Yes" 
    - "Final Answer:\n2. No"
    - "Therefore, Yes"
    - "The correct answer is:\n1. Yes"
    - "**1. Yes**"
    - "Choice 2: No"
    """
    if not response:
        return ''

    response_lower = response.lower()
    
    # Strong patterns - including numbered answers like "1. Yes" or "2. No"
    # Order matters - more specific patterns first
    patterns = [
        # "My answer is: Yes" or "My answer is: **Yes**" or "My answer is: **No**."
        r'[Mm]y answer is[:\s]*\*?\*?(?:\d+[.\)]\s*)?([Yy]es|[Nn]o)\b',
        # "**Final Answer: 2. No**" or "Final Answer:\n1. Yes"
        r'[Ff]inal [Aa]nswer[:\s]*\*?\*?(?:\d+[.\)]\s*)?([Yy]es|[Nn]o)\b',
        r'\*\*[Ff]inal [Aa]nswer[:\s]*(?:\d+[.\)]\s*)?([Yy]es|[Nn]o)\b',
        # "The correct answer is:\n1. Yes" or "correct answer is: No"
        r'correct answer (?:is|would be)[:\s]*\*?\*?(?:\d+[.\)]\s*)?([Yy]es|[Nn]o)\b',
        # "**Answer:**\n1. Yes" or "Answer: No"
        r'\*\*[Aa]nswer\*?\*?[:\s]*\n?\s*(?:\d+[.\)]\s*)?([Yy]es|[Nn]o)\b',
        r'[Aa]nswer[:\s]+\*?\*?(?:\d+[.\)]\s*)?([Yy]es|[Nn]o)\b',
        # "**1. Yes**" or "**2. No**" (bold numbered)
        r'\*\*\d+[.\)]\s*([Yy]es|[Nn]o)\*\*',
        # "1. Yes" or "2. No" standalone on line (not bold)
        r'(?:^|\n)\s*\d+[.\)]\s*([Yy]es|[Nn]o)\s*(?:\n|$)',
        # "Choice 1: Yes" or "Choice 2 (No)"
        r'[Cc]hoice\s*\d+[:\s]*\(?([Yy]es|[Nn]o)\)?',
        # "**Yes**" or "**No**" standalone
        r'\*\*([Yy]es|[Nn]o)\*\*',
        # "Therefore, yes" or "Thus, no"
        r'[Tt]herefore[,:\s]+(?:the answer (?:is|would be)[:\s]*)?([Yy]es|[Nn]o)\b',
        r'[Tt]hus[,:\s]+(?:the answer (?:is|would be)[:\s]*)?([Yy]es|[Nn]o)\b',
        # "Conclusion:\n...\nNo" - check conclusion section
        r'[Cc]onclusion[:\s]*\n?.*?([Yy]es|[Nn]o)\b',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_lower, re.MULTILINE | re.DOTALL)
        if match:
            return match.group(1).capitalize()
    
    # Check for numbered format in last few lines (e.g., "1. Yes" or "2. No")
    lines = response_lower.strip().split('\n')
    for line in reversed(lines[-10:]):
        line_stripped = line.strip()
        # Match patterns like "1. Yes", "1) Yes", "1.yes", etc.
        match = re.match(r'^\s*\d+[.\)]\s*([Yy]es|[Nn]o)\b', line_stripped)
        if match:
            return match.group(1).capitalize()
    
    # Weak check in last 100 chars
    last_part = response_lower[-100:]
    if 'yes' in last_part and 'no' not in last_part:
        return 'Yes'
    elif 'no' in last_part and 'yes' not in last_part:
        return 'No'
    
    return ''


def check_answer(predicted: str, gold: str) -> bool:
    """Check if predicted answer matches gold answer."""
    if not predicted or not gold:
        return False
    
    predicted = str(predicted).strip().lower()
    gold = str(gold).strip().lower()
    
    yes_variants = {'yes', 'true', 'correct', 'y', 'positive'}
    no_variants = {'no', 'false', 'incorrect', 'n', 'negative'}
    
    pred_is_yes = predicted in yes_variants
    pred_is_no = predicted in no_variants
    gold_is_yes = gold in yes_variants
    gold_is_no = gold in no_variants
    
    if (pred_is_yes or pred_is_no) and (gold_is_yes or gold_is_no):
        return pred_is_yes == gold_is_yes
    
    return predicted == gold


def majority_vote(answers: List[str]) -> str:
    """
    Get majority vote from list of answers.
    
    - Filters out empty answers before voting
    - Returns empty string if Yes and No counts are equal (tie)
    """
    if not answers:
        return ''
    
    # Normalize answers and filter out empty ones
    normalized = []
    for ans in answers:
        ans_str = str(ans).strip()
        if not ans_str:  # Skip empty answers
            continue
        ans_lower = ans_str.lower()
        if ans_lower in {'yes', 'true', 'correct', 'y', 'positive'}:
            normalized.append('Yes')
        elif ans_lower in {'no', 'false', 'incorrect', 'n', 'negative'}:
            normalized.append('No')
        else:
            normalized.append(ans_str)
    
    # If all answers were empty, return empty
    if not normalized:
        return ''
    
    # Count occurrences
    counter = Counter(normalized)
    
    # Check for tie between Yes and No
    yes_count = counter.get('Yes', 0)
    no_count = counter.get('No', 0)
    
    # If Yes and No are tied (and they are the top answers), return empty
    if yes_count > 0 and no_count > 0 and yes_count == no_count:
        # Check if they are the most common
        most_common = counter.most_common(2)
        if len(most_common) >= 2:
            if most_common[0][1] == most_common[1][1]:
                # Tie at the top - return empty
                return ''
    
    # Return most common answer
    if counter:
        return counter.most_common(1)[0][0]
    return ''


def run_single_sample(
    baseline,
    sample,
    iterations: int = 1,
) -> Dict:
    """Run a single sample with optional multiple iterations."""
    all_answers = []
    all_explanations = []
    
    for i in range(iterations):
        try:
            result = baseline.generate(
                context=sample.context,
                question=sample.question,
                choices=sample.choices
            )
            
            predicted = result.get('answer', '')
            if not predicted:
                predicted = extract_answer_from_response(result.get('explanation', ''))
            
            all_answers.append(predicted)
            all_explanations.append(result.get('explanation', ''))
            
        except Exception as e:
            logger.warning(f"Iteration {i+1} error: {e}")
            all_answers.append('')
            all_explanations.append(f"Error: {e}")
    
    # Get majority vote
    final_answer = majority_vote(all_answers) if iterations > 1 else (all_answers[0] if all_answers else '')
    is_correct = check_answer(final_answer, sample.answer)
    
    return {
        'sample_id': sample.id,
        'question': sample.question[:100],
        'gold_answer': sample.answer,
        'predicted_answer': final_answer,
        'is_correct': is_correct,
        'iterations': iterations,
        'all_answers': all_answers,
        'all_explanations': all_explanations,  # Record all explanations from all iterations
        'answer_distribution': dict(Counter(all_answers)) if iterations > 1 else None,
        'explanation': all_explanations[0] if all_explanations else '',  # Keep first for backward compatibility
    }


def run_parallel(baseline, samples, parallel_workers: int, iterations: int = 1) -> List[Dict]:
    """Run experiments in parallel using ThreadPoolExecutor."""
    results = []
    completed = 0
    total = len(samples)
    lock = Lock()
    
    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        futures = {
            executor.submit(run_single_sample, baseline, sample, iterations): idx 
            for idx, sample in enumerate(samples)
        }
        
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                if result:
                    with lock:
                        results.append(result)
                        completed += 1
                        if completed % 20 == 0 or completed == total:
                            logger.info(f"  Progress: {completed}/{total} samples completed")
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {e}")
    
    results.sort(key=lambda x: x.get('sample_id', ''))
    return results


def run_experiment(
    model_name: str,
    method_name: str,
    domain: str,
    dataset_name: str,
    output_dir: Path,
    server_url: str = 'http://localhost:8000/v1',
    max_samples: int = None,
    iterations: int = 1,
    parallel_workers: int = 4,
    skip_existing: bool = True,
):
    """Run a single experiment with optional iterations and parallel workers."""
    # Check if result already exists
    if skip_existing:
        iter_suffix = f"_iter{iterations}" if iterations > 1 else ""
        output_file = output_dir / f"{model_name}_{method_name}_{domain}_{dataset_name}{iter_suffix}.json"
        if output_file.exists():
            logger.info(f"Skipping (result exists): {output_file}")
            # Load existing result to return summary
            try:
                with open(output_file, 'r') as f:
                    existing_data = json.load(f)
                return {
                    'model': model_name,
                    'method': method_name,
                    'domain': domain,
                    'dataset': dataset_name,
                    'accuracy': existing_data.get('accuracy', 0),
                    'n_samples': existing_data.get('total', 0),
                    'iterations': iterations,
                    'skipped': True,
                }
            except Exception:
                pass  # If loading fails, re-run the experiment
    
    logger.info(f"Running {model_name}/{method_name}/{domain}/{dataset_name} (iterations={iterations}, workers={parallel_workers})")
    
    # Initialize client
    try:
        client = get_client(model_name, use_server=True, server_url=server_url)
    except Exception as e:
        logger.error(f"Failed to initialize client: {e}")
        return None
    
    # Initialize baseline
    baseline = get_baseline(method_name, llm_client=client)
    
    # Load data
    loader = MultiDomainLoader()
    samples = loader.load(domain, dataset_name, max_samples=max_samples)
    
    if not samples:
        logger.warning(f"No samples loaded for {domain}/{dataset_name}")
        return None
    
    logger.info(f"Loaded {len(samples)} samples")
    
    # Run experiments
    results = []
    correct = 0
    
    if parallel_workers > 1 and iterations == 1:
        # Parallel execution only for single iteration (thread safety)
        results = run_parallel(baseline, samples, parallel_workers, iterations)
        correct = sum(1 for r in results if r['is_correct'])
    else:
        # Sequential execution
        for idx, sample in enumerate(tqdm(samples, desc=f"{method_name}/{dataset_name}")):
            result = run_single_sample(baseline, sample, iterations)
            results.append(result)
            if result['is_correct']:
                correct += 1
    
    # Calculate accuracy
    accuracy = correct / len(results) if results else 0
    logger.info(f"Accuracy: {accuracy:.1%} ({correct}/{len(results)})")
    
    # Save results
    iter_suffix = f"_iter{iterations}" if iterations > 1 else ""
    output_file = output_dir / f"{model_name}_{method_name}_{domain}_{dataset_name}{iter_suffix}.json"
    
    output_data = {
        'model': model_name,
        'method': method_name,
        'domain': domain,
        'dataset': dataset_name,
        'accuracy': accuracy,
        'correct': correct,
        'total': len(results),
        'iterations': iterations,
        'majority_voting': iterations > 1,
        'timestamp': datetime.now().isoformat(),
        'results': results,
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Saved to {output_file}")
    
    return {
        'model': model_name,
        'method': method_name,
        'domain': domain,
        'dataset': dataset_name,
        'accuracy': accuracy,
        'n_samples': len(results),
        'iterations': iterations,
    }


def main():
    parser = argparse.ArgumentParser(description='Run SEF baseline experiments')
    parser.add_argument('--model', type=str, choices=MODELS, required=True,
                       help='Model to evaluate')
    parser.add_argument('--method', type=str, choices=METHODS,
                       help='Method to run (default: all)')
    parser.add_argument('--domain', type=str, choices=DOMAINS + ['all'], default='all',
                       help='Domain to run (default: all)')
    parser.add_argument('--dataset', type=str,
                       help='Specific dataset to run (default: all in domain)')
    parser.add_argument('--output-dir', type=str, default='results/raw',
                       help='Output directory')
    parser.add_argument('--server-url', type=str, default='http://localhost:8000/v1',
                       help='vLLM server URL')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Max samples per dataset (for testing)')
    parser.add_argument('--iterations', type=int, default=1,
                       help='Number of iterations for majority voting (default: 1)')
    parser.add_argument('--parallel-workers', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                       help='Skip experiments with existing results (default: True)')
    parser.add_argument('--no-skip-existing', dest='skip_existing', action='store_false',
                       help='Re-run experiments even if results exist')
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    methods = [args.method] if args.method else METHODS
    domains = DOMAINS if args.domain == 'all' else [args.domain]
    
    loader = MultiDomainLoader()
    all_results = []
    
    for domain in domains:
        datasets_in_domain = loader.get_available_datasets(domain)[domain]
        
        if args.dataset:
            if args.dataset in datasets_in_domain:
                datasets_to_run = [args.dataset]
            else:
                logger.warning(f"Dataset {args.dataset} not found in {domain}")
                continue
        else:
            datasets_to_run = datasets_in_domain
        
        for dataset_name in datasets_to_run:
            for method in methods:
                try:
                    result = run_experiment(
                        model_name=args.model,
                        method_name=method,
                        domain=domain,
                        dataset_name=dataset_name,
                        output_dir=output_dir,
                        server_url=args.server_url,
                        max_samples=args.max_samples,
                        iterations=args.iterations,
                        parallel_workers=args.parallel_workers,
                        skip_existing=args.skip_existing,
                    )
                    if result:
                        all_results.append(result)
                except Exception as e:
                    logger.error(f"Failed {args.model}/{method}/{domain}/{dataset_name}: {e}")
    
    # Print summary
    print("\n" + "=" * 90)
    print(f"BASELINE EXPERIMENT SUMMARY (iterations={args.iterations})")
    print("=" * 90)
    print(f"{'Model':<15} {'Method':<15} {'Domain':<12} {'Dataset':<25} {'Accuracy':<10}")
    print("-" * 90)
    
    for result in all_results:
        print(f"{result['model']:<15} {result['method']:<15} {result['domain']:<12} "
              f"{result['dataset']:<25} {result['accuracy']:.4f}")
    
    print("=" * 90)


if __name__ == "__main__":
    main()
