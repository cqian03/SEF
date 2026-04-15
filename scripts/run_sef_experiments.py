#!/usr/bin/env python3
"""
Run SEF (Structured Explanation Framework) experiments across all domains.

Supports iterative evaluation with majority voting.

Usage:
    # Run SEF with 5 iterations (majority voting)
    python scripts/run_sef_experiments.py --model qwen_2_5_14b --domain all --iterations 5
    
    # Run specific ablation
    python scripts/run_sef_experiments.py --model qwen_2_5_14b --ablation no_afl --iterations 5
    
    # Run all ablations
    python scripts/run_sef_experiments.py --model qwen_2_5_14b --all-ablations --iterations 5
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
from typing import List, Dict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from src.baselines.sef_prompting import SEFPrompting, SEFAblation
from src.llm_clients import get_client
from src.data_loader import MultiDomainLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_answer_from_explanation(explanation: str) -> str:
    """
    Extract answer from explanation text.
    
    Handles various formats including:
    - "My answer is: Yes" or "My answer is: **Yes**."
    - "**Answer:**\n1. Yes"
    - "Final Answer:\n2. No"
    - "The correct answer is:\n1. Yes"
    - "**1. Yes**"
    - "Choice 2: No"
    """
    if not explanation:
        return ''
    
    explanation_lower = explanation.lower()
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
        match = re.search(pattern, explanation_lower, re.MULTILINE | re.DOTALL)
        if match:
            return match.group(1).capitalize()
    
    # Check for numbered format in last few lines (e.g., "1. Yes" or "2. No")
    lines = explanation_lower.strip().split('\n')
    for line in reversed(lines[-10:]):
        line_stripped = line.strip()
        # Match patterns like "1. Yes", "1) Yes", "1.yes", etc.
        match = re.match(r'^\s*\d+[.\)]\s*([Yy]es|[Nn]o)\b', line_stripped)
        if match:
            return match.group(1).capitalize()
    
    # Weak check in last 100 chars
    last_part = explanation_lower[-100:]
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


# Models used in the paper (Section 4.1)
MODELS = [
    'deepseek_r1_14b',
    'gemma_3_12b',
    'ministral_14b',
    'qwen_2_5_14b',
]

DOMAINS = ['legal', 'medical', 'financial']

# Full SEF (no ablation)
FULL_SEF = 'full'

# Ablation types (each removes one component)
ABLATION_TYPES = [
    'no_afl',           # Remove Answer First/Last
    'no_ac',            # Remove Answer Clarity
    'no_ci',            # Remove Conclusion Isolation
    'no_dtc',           # Remove Domain Terminology Consistency
    'no_cea',           # Remove Context-Evidence Alignment
    'no_fs',            # Remove Fact Specificity
    'no_presentation',  # Remove all presentation metrics (AFL, AC, CI)
    'no_domain',        # Remove all domain metrics (DTC, CEA, FS)
]


def process_single_sample(method, sample, sample_idx, domain, iterations: int = 1) -> Dict:
    """Process a single sample with optional multiple iterations."""
    all_answers = []
    all_explanations = []
    
    for i in range(iterations):
        try:
            result = method.generate(
                context=sample.context,
                question=sample.question,
                choices=sample.choices,
                domain=domain,
            )
            
            explanation = result.get('explanation', '')
            predicted_answer = extract_answer_from_explanation(explanation)
            
            if not predicted_answer:
                predicted_answer = result.get('answer', '')
            
            all_answers.append(predicted_answer)
            all_explanations.append(explanation)
            
        except Exception as e:
            logger.warning(f"Sample {sample_idx}, iteration {i+1} error: {e}")
            all_answers.append('')
            all_explanations.append(f"Error: {e}")
    
    # Get majority vote
    final_answer = majority_vote(all_answers) if iterations > 1 else (all_answers[0] if all_answers else '')
    is_correct = check_answer(final_answer, sample.answer)
    
    return {
        'sample_id': sample.id,
        'predicted_answer': final_answer,
        'gold_answer': sample.answer,
        'is_correct': is_correct,
        'iterations': iterations,
        'all_answers': all_answers,
        'all_explanations': all_explanations,  # Record all explanations from all iterations
        'answer_distribution': dict(Counter(all_answers)) if iterations > 1 else None,
        'explanation': all_explanations[0] if all_explanations else '',  # Keep first for backward compatibility
    }


def run_experiment(
    model_name: str,
    domain: str,
    dataset_name: str,
    ablation_type: str = 'full',
    max_samples: int = None,
    output_dir: Path = None,
    server_url: str = 'http://localhost:8000/v1',
    parallel_workers: int = 1,
    iterations: int = 1,
    skip_existing: bool = True,
):
    """Run a single SEF experiment with optional iterations."""
    # Check if result already exists
    if output_dir and skip_existing:
        method_name = f"sef_{ablation_type}" if ablation_type != 'full' else 'sef'
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
                    'domain': domain,
                    'dataset': dataset_name,
                    'ablation': ablation_type,
                    'accuracy': existing_data.get('accuracy', 0),
                    'n_samples': existing_data.get('n_samples', 0),
                    'iterations': iterations,
                    'skipped': True,
                }
            except Exception:
                pass  # If loading fails, re-run the experiment
    
    logger.info(f"Running SEF ({ablation_type}) on {model_name} / {domain} / {dataset_name} (iterations={iterations})")
    
    # Initialize client
    try:
        llm_client = get_client(
            provider=model_name,
            use_server=True,
            server_url=server_url,
        )
    except Exception as e:
        logger.error(f"Failed to initialize client for {model_name}: {e}")
        return None
    
    # Initialize SEF method
    if ablation_type == 'full':
        method = SEFPrompting(llm_client, domain=domain)
    else:
        method = SEFAblation(llm_client, ablation_type=ablation_type, domain=domain)
    
    # Load data
    data_loader = MultiDomainLoader()
    samples = data_loader.load(domain, dataset_name, max_samples=max_samples)
    
    if not samples:
        logger.warning(f"No samples loaded for {domain}/{dataset_name}")
        return None
    
    logger.info(f"Loaded {len(samples)} samples")
    
    # Run experiments
    results = []
    
    if parallel_workers > 1 and iterations == 1:
        # Parallel execution only for single iteration (thread safety)
        results = run_parallel(method, samples, parallel_workers, domain, iterations)
    else:
        # Sequential execution
        for idx, sample in enumerate(tqdm(samples, desc=f"SEF/{dataset_name}")):
            result = process_single_sample(method, sample, idx, domain, iterations)
            if result:
                results.append(result)
    
    # Calculate accuracy
    accuracy = sum(r['is_correct'] for r in results) / len(results) if results else 0
    logger.info(f"Accuracy: {accuracy:.4f} ({sum(r['is_correct'] for r in results)}/{len(results)})")
    
    # Save results
    if output_dir:
        method_name = f"sef_{ablation_type}" if ablation_type != 'full' else 'sef'
        iter_suffix = f"_iter{iterations}" if iterations > 1 else ""
        output_file = output_dir / f"{model_name}_{method_name}_{domain}_{dataset_name}{iter_suffix}.json"
        
        output_data = {
            'model': model_name,
            'method': method_name,
            'domain': domain,
            'dataset': dataset_name,
            'ablation': ablation_type,
            'accuracy': accuracy,
            'n_samples': len(results),
            'iterations': iterations,
            'majority_voting': iterations > 1,
            'timestamp': datetime.now().isoformat(),
            'results': results,
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Saved results to {output_file}")
    
    return {
        'model': model_name,
        'domain': domain,
        'dataset': dataset_name,
        'ablation': ablation_type,
        'accuracy': accuracy,
        'n_samples': len(results),
        'iterations': iterations,
    }


def run_parallel(method, samples, parallel_workers, domain, iterations):
    """Run experiments in parallel using ThreadPoolExecutor."""
    results = []
    completed = 0
    total = len(samples)
    lock = Lock()
    
    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        futures = {
            executor.submit(process_single_sample, method, sample, idx, domain, iterations): idx 
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


def main():
    parser = argparse.ArgumentParser(description='Run SEF experiments')
    parser.add_argument('--model', type=str, required=True, 
                       choices=MODELS, help='Model to use')
    parser.add_argument('--domain', type=str, default='all',
                       choices=DOMAINS + ['all'], help='Domain to run (default: all)')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Specific dataset to run (default: all in domain)')
    parser.add_argument('--ablation', type=str, default='full',
                       choices=['full'] + ABLATION_TYPES, help='Ablation type (default: full)')
    parser.add_argument('--all-ablations', action='store_true',
                       help='Run all ablation types')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples per dataset')
    parser.add_argument('--output-dir', type=str, default='results/raw',
                       help='Output directory')
    parser.add_argument('--server-url', type=str, default='http://localhost:8000/v1',
                       help='vLLM server URL')
    parser.add_argument('--parallel-workers', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    parser.add_argument('--iterations', type=int, default=1,
                       help='Number of iterations for majority voting (default: 1)')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                       help='Skip experiments with existing results (default: True)')
    parser.add_argument('--no-skip-existing', dest='skip_existing', action='store_false',
                       help='Re-run experiments even if results exist')
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    domains = DOMAINS if args.domain == 'all' else [args.domain]
    ablations = ABLATION_TYPES if args.all_ablations else [args.ablation]
    
    loader = MultiDomainLoader()
    all_results = []
    
    for ablation in ablations:
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
                result = run_experiment(
                    model_name=args.model,
                    domain=domain,
                    dataset_name=dataset_name,
                    ablation_type=ablation,
                    max_samples=args.max_samples,
                    output_dir=output_dir,
                    server_url=args.server_url,
                    parallel_workers=args.parallel_workers,
                    iterations=args.iterations,
                    skip_existing=args.skip_existing,
                )
                if result:
                    all_results.append(result)
    
    # Print summary
    print("\n" + "=" * 100)
    print(f"SEF EXPERIMENT SUMMARY (iterations={args.iterations})")
    print("=" * 100)
    print(f"{'Model':<15} {'Ablation':<18} {'Domain':<12} {'Dataset':<25} {'Accuracy':<10}")
    print("-" * 100)
    
    for result in all_results:
        print(f"{result['model']:<15} {result['ablation']:<18} {result['domain']:<12} "
              f"{result['dataset']:<25} {result['accuracy']:.4f}")
    
    print("=" * 100)


if __name__ == '__main__':
    main()
