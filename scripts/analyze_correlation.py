#!/usr/bin/env python3
"""
Analyze correlations between SEF metrics and accuracy.

Computes correlations at multiple levels:
1. Per-sample: Metric scores vs correctness (binary)
2. Per-experiment: Aggregated metric scores vs accuracy
3. By method/model/domain: Grouped correlation analysis

Uses both Pearson and Spearman correlations.

Usage:
    python scripts/analyze_correlation.py --metrics-dir results/metrics
    python scripts/analyze_correlation.py --results-dir results/raw --compute-metrics
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import scipy for correlation computation
try:
    from scipy import stats
    import numpy as np
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available. Using basic correlation computation.")


# Display names
METRIC_DISPLAY = {
    'AFL': 'Answer First/Last',
    'AC': 'Answer Clarity',
    'CI': 'Conclusion Isolation',
    'DTC': 'Domain Term. Consist.',
    'CEA': 'Conclusion-Evidence Align.',
    'FS': 'Fact Specificity',
}

METHOD_DISPLAY = {
    'direct': 'Direct',
    'standard_cot': 'CoT',
    'tree_of_thought': 'ToT',
    'chain_of_verification': 'CoVe',
    'vanilla_rag': 'V-RAG',
    'self_rag': 'Self-RAG',
    'sef': 'SEF (Full)',
}


def compute_pearson(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Compute Pearson correlation coefficient and p-value."""
    if SCIPY_AVAILABLE:
        return stats.pearsonr(x, y)
    else:
        # Simple Pearson correlation without p-value
        n = len(x)
        if n < 3:
            return 0.0, 1.0
        
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        denom_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
        denom_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5
        
        if denom_x == 0 or denom_y == 0:
            return 0.0, 1.0
        
        r = numerator / (denom_x * denom_y)
        return r, 0.0  # p-value not computed without scipy


def compute_spearman(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Compute Spearman rank correlation coefficient and p-value."""
    if SCIPY_AVAILABLE:
        return stats.spearmanr(x, y)
    else:
        # Rank-based correlation
        n = len(x)
        if n < 3:
            return 0.0, 1.0
        
        # Simple ranking (doesn't handle ties well)
        rank_x = [sorted(x).index(xi) + 1 for xi in x]
        rank_y = [sorted(y).index(yi) + 1 for yi in y]
        
        return compute_pearson(rank_x, rank_y)


def load_metrics_files(metrics_dir: Path) -> List[Dict]:
    """Load all metrics files."""
    results = []
    
    for filepath in metrics_dir.glob('metrics_*.json'):
        try:
            with open(filepath) as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            logger.warning(f"Failed to load {filepath}: {e}")
    
    logger.info(f"Loaded {len(results)} metrics files")
    return results


def compute_per_sample_correlations(
    all_results: List[Dict]
) -> Dict[str, Dict[str, Any]]:
    """
    Compute per-sample correlations between metrics and correctness.
    
    This correlates individual sample metric scores with binary correctness.
    High statistical power due to large sample size.
    """
    # Collect all samples
    all_samples = []
    for result in all_results:
        for sample in result.get('sample_metrics', []):
            all_samples.append({
                'is_correct': 1 if sample['is_correct'] else 0,
                **sample['metrics']
            })
    
    if len(all_samples) < 10:
        logger.warning("Not enough samples for per-sample correlation")
        return {}
    
    correlations = {}
    correctness = [s['is_correct'] for s in all_samples]
    
    for metric_name in ['AFL', 'AC', 'CI', 'DTC', 'CEA', 'FS']:
        scores = [s[metric_name] for s in all_samples]
        
        # Pearson correlation
        r_pearson, p_pearson = compute_pearson(scores, correctness)
        
        # Spearman correlation
        r_spearman, p_spearman = compute_spearman(scores, correctness)
        
        # Effect size (mean difference)
        correct_scores = [s[metric_name] for s in all_samples if s['is_correct'] == 1]
        incorrect_scores = [s[metric_name] for s in all_samples if s['is_correct'] == 0]
        
        mean_correct = sum(correct_scores) / len(correct_scores) if correct_scores else 0
        mean_incorrect = sum(incorrect_scores) / len(incorrect_scores) if incorrect_scores else 0
        
        correlations[metric_name] = {
            'pearson_r': round(r_pearson, 4),
            'pearson_p': p_pearson,
            'spearman_r': round(r_spearman, 4),
            'spearman_p': p_spearman,
            'n_samples': len(scores),
            'mean_correct': round(mean_correct, 4),
            'mean_incorrect': round(mean_incorrect, 4),
            'effect_size': round(mean_correct - mean_incorrect, 4)
        }
    
    return correlations


def compute_experiment_correlations(
    all_results: List[Dict]
) -> Dict[str, Dict[str, Any]]:
    """
    Compute experiment-level correlations.
    
    This correlates aggregated metric scores per experiment with accuracy.
    More meaningful for understanding method effectiveness.
    """
    if len(all_results) < 5:
        logger.warning("Not enough experiments for correlation")
        return {}
    
    correlations = {}
    accuracies = [r['accuracy'] for r in all_results]
    
    for metric_name in ['AFL', 'AC', 'CI', 'DTC', 'CEA', 'FS']:
        scores = [r['aggregated_metrics'][metric_name] for r in all_results]
        
        # Check for variance
        if SCIPY_AVAILABLE:
            if np.std(scores) == 0 or np.std(accuracies) == 0:
                correlations[metric_name] = {
                    'pearson_r': 0.0,
                    'pearson_p': 1.0,
                    'spearman_r': 0.0,
                    'spearman_p': 1.0,
                    'n_experiments': len(all_results)
                }
                continue
        
        r_pearson, p_pearson = compute_pearson(scores, accuracies)
        r_spearman, p_spearman = compute_spearman(scores, accuracies)
        
        correlations[metric_name] = {
            'pearson_r': round(r_pearson, 4),
            'pearson_p': p_pearson,
            'spearman_r': round(r_spearman, 4),
            'spearman_p': p_spearman,
            'n_experiments': len(all_results)
        }
    
    return correlations


def compute_grouped_correlations(
    all_results: List[Dict],
    group_by: str = 'method'
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Compute correlations grouped by method, model, or domain.
    """
    # Group results
    grouped = defaultdict(list)
    for result in all_results:
        key = result.get(group_by, 'unknown')
        grouped[key].append(result)
    
    group_correlations = {}
    
    for group_name, group_results in grouped.items():
        if len(group_results) < 3:
            continue
        
        # Per-sample correlations for this group
        all_samples = []
        for result in group_results:
            for sample in result.get('sample_metrics', []):
                all_samples.append({
                    'is_correct': 1 if sample['is_correct'] else 0,
                    **sample['metrics']
                })
        
        if len(all_samples) < 10:
            continue
        
        correlations = {}
        correctness = [s['is_correct'] for s in all_samples]
        
        for metric_name in ['AFL', 'AC', 'CI', 'DTC', 'CEA', 'FS']:
            scores = [s[metric_name] for s in all_samples]
            r, p = compute_pearson(scores, correctness)
            
            correlations[metric_name] = {
                'pearson_r': round(r, 4),
                'pearson_p': p,
                'n_samples': len(scores)
            }
        
        group_correlations[group_name] = correlations
    
    return group_correlations


def generate_correlation_report(
    per_sample_corr: Dict,
    experiment_corr: Dict,
    method_corr: Dict,
    domain_corr: Dict,
) -> str:
    """Generate a comprehensive correlation report."""
    lines = []
    
    lines.append("=" * 80)
    lines.append("SEF METRICS CORRELATION ANALYSIS")
    lines.append("=" * 80)
    lines.append("")
    
    # ============================================
    # 1. Per-Sample Correlations (Main Results)
    # ============================================
    if per_sample_corr:
        n_samples = per_sample_corr.get('AFL', {}).get('n_samples', 0)
        lines.append("-" * 80)
        lines.append("1. PER-SAMPLE CORRELATIONS (Metrics vs Correctness)")
        lines.append("-" * 80)
        lines.append("")
        lines.append(f"n = {n_samples} samples")
        lines.append("")
        
        # Presentation dimension
        lines.append("Presentation Dimension:")
        for metric_name in ['AFL', 'AC', 'CI']:
            corr = per_sample_corr.get(metric_name, {})
            r = corr.get('pearson_r', 0)
            p = corr.get('pearson_p', 1)
            effect = corr.get('effect_size', 0)
            
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
            display = METRIC_DISPLAY.get(metric_name, metric_name)
            lines.append(f"  {display:<28}: r = {r:>7.4f}{sig:<4} (effect = {effect:>+.4f})")
        
        lines.append("")
        lines.append("Domain Reasoning Dimension:")
        for metric_name in ['DTC', 'CEA', 'FS']:
            corr = per_sample_corr.get(metric_name, {})
            r = corr.get('pearson_r', 0)
            p = corr.get('pearson_p', 1)
            effect = corr.get('effect_size', 0)
            
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
            display = METRIC_DISPLAY.get(metric_name, metric_name)
            lines.append(f"  {display:<28}: r = {r:>7.4f}{sig:<4} (effect = {effect:>+.4f})")
        
        lines.append("")
        lines.append("Significance: * p < 0.05, ** p < 0.01, *** p < 0.001")
        lines.append("")
    
    # ============================================
    # 2. Experiment-Level Correlations
    # ============================================
    if experiment_corr:
        n_exp = experiment_corr.get('AFL', {}).get('n_experiments', 0)
        lines.append("-" * 80)
        lines.append("2. EXPERIMENT-LEVEL CORRELATIONS (Avg Metrics vs Accuracy)")
        lines.append("-" * 80)
        lines.append("")
        lines.append(f"n = {n_exp} experiments")
        lines.append("")
        
        header = f"{'Metric':<30} {'Pearson r':<12} {'Spearman r':<12}"
        lines.append(header)
        lines.append("-" * 54)
        
        for metric_name in ['AFL', 'AC', 'CI', 'DTC', 'CEA', 'FS']:
            corr = experiment_corr.get(metric_name, {})
            r_p = corr.get('pearson_r', 0)
            r_s = corr.get('spearman_r', 0)
            display = METRIC_DISPLAY.get(metric_name, metric_name)
            lines.append(f"  {display:<28} {r_p:>10.4f}   {r_s:>10.4f}")
        
        lines.append("")
    
    # ============================================
    # 3. Correlations by Method
    # ============================================
    if method_corr:
        lines.append("-" * 80)
        lines.append("3. CORRELATIONS BY METHOD")
        lines.append("-" * 80)
        lines.append("")
        
        for method, correlations in sorted(method_corr.items()):
            display = METHOD_DISPLAY.get(method, method)
            n = correlations.get('AFL', {}).get('n_samples', 0)
            lines.append(f"{display} (n={n}):")
            
            for metric_name in ['AFL', 'AC', 'CI', 'DTC', 'CEA', 'FS']:
                corr = correlations.get(metric_name, {})
                r = corr.get('pearson_r', 0)
                p = corr.get('pearson_p', 1)
                sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
                lines.append(f"    {metric_name}: r = {r:>7.4f}{sig}")
            
            lines.append("")
    
    # ============================================
    # 4. Correlations by Domain
    # ============================================
    if domain_corr:
        lines.append("-" * 80)
        lines.append("4. CORRELATIONS BY DOMAIN")
        lines.append("-" * 80)
        lines.append("")
        
        for domain, correlations in sorted(domain_corr.items()):
            n = correlations.get('AFL', {}).get('n_samples', 0)
            lines.append(f"{domain.title()} Domain (n={n}):")
            
            for metric_name in ['AFL', 'AC', 'CI', 'DTC', 'CEA', 'FS']:
                corr = correlations.get(metric_name, {})
                r = corr.get('pearson_r', 0)
                p = corr.get('pearson_p', 1)
                sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
                lines.append(f"    {metric_name}: r = {r:>7.4f}{sig}")
            
            lines.append("")
    
    # ============================================
    # Summary Statistics
    # ============================================
    lines.append("-" * 80)
    lines.append("SUMMARY")
    lines.append("-" * 80)
    lines.append("")
    
    if per_sample_corr:
        # Find strongest correlations
        sorted_metrics = sorted(
            per_sample_corr.items(),
            key=lambda x: abs(x[1].get('pearson_r', 0)),
            reverse=True
        )
        
        lines.append("Strongest metric correlations with correctness:")
        for metric, corr in sorted_metrics[:3]:
            r = corr.get('pearson_r', 0)
            display = METRIC_DISPLAY.get(metric, metric)
            lines.append(f"  1. {display}: r = {r:.4f}")
        
        lines.append("")
        
        # Average correlation by dimension
        presentation_r = sum(
            per_sample_corr.get(m, {}).get('pearson_r', 0) 
            for m in ['AFL', 'AC', 'CI']
        ) / 3
        domain_r = sum(
            per_sample_corr.get(m, {}).get('pearson_r', 0) 
            for m in ['DTC', 'CEA', 'FS']
        ) / 3
        
        lines.append(f"Average Presentation correlation: r = {presentation_r:.4f}")
        lines.append(f"Average Domain Reasoning correlation: r = {domain_r:.4f}")
        lines.append("")
    
    lines.append("=" * 80)
    
    return '\n'.join(lines)


def generate_markdown_report(
    per_sample_corr: Dict,
    experiment_corr: Dict,
    method_corr: Dict,
    domain_corr: Dict,
) -> str:
    """Generate a markdown correlation report."""
    lines = []
    
    lines.append("# SEF Metrics Correlation Analysis")
    lines.append("")
    
    # 1. Per-Sample Correlations
    if per_sample_corr:
        n_samples = per_sample_corr.get('AFL', {}).get('n_samples', 0)
        lines.append("## 1. Per-Sample Correlations")
        lines.append("")
        lines.append(f"Correlations between metric scores and correctness (n = {n_samples} samples)")
        lines.append("")
        lines.append("| Metric | Pearson r | p-value | Effect Size |")
        lines.append("|--------|-----------|---------|-------------|")
        
        for metric_name in ['AFL', 'AC', 'CI', 'DTC', 'CEA', 'FS']:
            corr = per_sample_corr.get(metric_name, {})
            r = corr.get('pearson_r', 0)
            p = corr.get('pearson_p', 1)
            effect = corr.get('effect_size', 0)
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
            
            p_str = f"{p:.4f}" if p >= 0.0001 else "<0.0001"
            lines.append(f"| {metric_name} | {r:.4f}{sig} | {p_str} | {effect:+.4f} |")
        
        lines.append("")
        lines.append("*Significance: \\* p < 0.05, \\*\\* p < 0.01, \\*\\*\\* p < 0.001*")
        lines.append("")
    
    # 2. Experiment-Level Correlations
    if experiment_corr:
        n_exp = experiment_corr.get('AFL', {}).get('n_experiments', 0)
        lines.append("## 2. Experiment-Level Correlations")
        lines.append("")
        lines.append(f"Correlations between average metric scores and accuracy (n = {n_exp} experiments)")
        lines.append("")
        lines.append("| Metric | Pearson r | Spearman r |")
        lines.append("|--------|-----------|------------|")
        
        for metric_name in ['AFL', 'AC', 'CI', 'DTC', 'CEA', 'FS']:
            corr = experiment_corr.get(metric_name, {})
            r_p = corr.get('pearson_r', 0)
            r_s = corr.get('spearman_r', 0)
            lines.append(f"| {metric_name} | {r_p:.4f} | {r_s:.4f} |")
        
        lines.append("")
    
    # 3. Correlations by Method
    if method_corr:
        lines.append("## 3. Correlations by Method")
        lines.append("")
        
        header = "| Method |"
        separator = "|--------|"
        for m in ['AFL', 'AC', 'CI', 'DTC', 'CEA', 'FS']:
            header += f" {m} |"
            separator += "------|"
        
        lines.append(header)
        lines.append(separator)
        
        for method, correlations in sorted(method_corr.items()):
            display = METHOD_DISPLAY.get(method, method)
            row = f"| {display} |"
            for metric_name in ['AFL', 'AC', 'CI', 'DTC', 'CEA', 'FS']:
                r = correlations.get(metric_name, {}).get('pearson_r', 0)
                row += f" {r:.3f} |"
            lines.append(row)
        
        lines.append("")
    
    # 4. Correlations by Domain
    if domain_corr:
        lines.append("## 4. Correlations by Domain")
        lines.append("")
        
        header = "| Domain |"
        separator = "|--------|"
        for m in ['AFL', 'AC', 'CI', 'DTC', 'CEA', 'FS']:
            header += f" {m} |"
            separator += "------|"
        
        lines.append(header)
        lines.append(separator)
        
        for domain, correlations in sorted(domain_corr.items()):
            row = f"| {domain.title()} |"
            for metric_name in ['AFL', 'AC', 'CI', 'DTC', 'CEA', 'FS']:
                r = correlations.get(metric_name, {}).get('pearson_r', 0)
                row += f" {r:.3f} |"
            lines.append(row)
        
        lines.append("")
    
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze correlations between SEF metrics and accuracy'
    )
    parser.add_argument(
        '--metrics-dir', type=str, default='results/metrics',
        help='Directory containing metrics files'
    )
    parser.add_argument(
        '--output-dir', type=str, default='results/correlations',
        help='Output directory for correlation analysis'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress stdout output'
    )
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    metrics_dir = project_root / args.metrics_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not metrics_dir.exists():
        logger.error(f"Metrics directory not found: {metrics_dir}")
        logger.info("Run analyze_metrics.py first to generate metrics files")
        return
    
    # Load metrics files
    all_results = load_metrics_files(metrics_dir)
    
    if not all_results:
        logger.error("No metrics files found!")
        return
    
    # Compute correlations
    logger.info("Computing per-sample correlations...")
    per_sample_corr = compute_per_sample_correlations(all_results)
    
    logger.info("Computing experiment-level correlations...")
    experiment_corr = compute_experiment_correlations(all_results)
    
    logger.info("Computing correlations by method...")
    method_corr = compute_grouped_correlations(all_results, 'method')
    
    logger.info("Computing correlations by domain...")
    domain_corr = compute_grouped_correlations(all_results, 'domain')
    
    # Generate reports
    text_report = generate_correlation_report(
        per_sample_corr, experiment_corr, method_corr, domain_corr
    )
    md_report = generate_markdown_report(
        per_sample_corr, experiment_corr, method_corr, domain_corr
    )
    
    if not args.quiet:
        print(text_report)
    
    # Save reports
    text_file = output_dir / "correlation_report.txt"
    with open(text_file, 'w') as f:
        f.write(text_report)
    logger.info(f"Text report saved to {text_file}")
    
    md_file = output_dir / "correlation_report.md"
    with open(md_file, 'w') as f:
        f.write(md_report)
    logger.info(f"Markdown report saved to {md_file}")
    
    # Save raw correlation data
    corr_data = {
        'per_sample': per_sample_corr,
        'experiment_level': experiment_corr,
        'by_method': method_corr,
        'by_domain': domain_corr,
    }
    
    json_file = output_dir / "correlations.json"
    with open(json_file, 'w') as f:
        json.dump(corr_data, f, indent=2)
    logger.info(f"Correlation data saved to {json_file}")


if __name__ == '__main__':
    main()

