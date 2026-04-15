#!/usr/bin/env python3
"""
Compute SEF metrics on experiment explanations.

SEF Metrics (6 total):
- Presentation Dimension (3 metrics):
  * AFL: Answer First/Last - Answer stated at beginning and/or end
  * AC: Answer Clarity - Clear, unambiguous answer statement
  * CI: Conclusion Isolation - Separate conclusion from reasoning
  
- Domain Reasoning Dimension (3 metrics):
  * DTC: Domain Terminology Consistency - Use of domain-specific terms
  * CEA: Conclusion Evidence Alignment - Conclusion grounded in evidence
  * FS: Fact Specificity - Use of specific facts vs vague statements

Usage:
    python scripts/analyze_metrics.py --results-dir results/raw
    python scripts/analyze_metrics.py --output-dir results/metrics
"""

import os
import sys
import json
import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Domain-specific terminology dictionaries
DOMAIN_TERMS = {
    'legal': [
        'hearsay', 'testimony', 'evidence', 'statute', 'precedent', 'court',
        'judge', 'jury', 'verdict', 'plaintiff', 'defendant', 'witness',
        'contract', 'clause', 'liability', 'negligence', 'tort', 'damages',
        'appeal', 'motion', 'ruling', 'jurisdiction', 'counsel', 'attorney',
        'prosecution', 'defense', 'objection', 'admissible', 'inadmissible',
        'burden of proof', 'reasonable doubt', 'breach', 'duty of care',
        'proximate cause', 'trespass', 'consideration', 'confidentiality',
    ],
    'medical': [
        'diagnosis', 'prognosis', 'symptoms', 'treatment', 'efficacy',
        'patient', 'clinical', 'therapy', 'disease', 'condition', 'medication',
        'dosage', 'side effects', 'adverse', 'contraindication', 'indication',
        'etiology', 'pathology', 'syndrome', 'chronic', 'acute', 'remission',
        'intervention', 'procedure', 'outcome', 'mortality', 'morbidity',
        'randomized', 'trial', 'placebo', 'cohort', 'statistical', 'significant',
    ],
    'financial': [
        'revenue', 'earnings', 'dividend', 'valuation', 'sentiment', 'stock',
        'market', 'investor', 'portfolio', 'asset', 'liability', 'equity',
        'profit', 'loss', 'margin', 'growth', 'decline', 'bullish', 'bearish',
        'positive', 'negative', 'neutral', 'forecast', 'analysis', 'trend',
        'volatility', 'risk', 'return', 'investment', 'capital', 'debt',
        'balance sheet', 'cash flow', 'operating', 'quarterly', 'annual',
    ],
}


class SEFMetrics:
    """Compute SEF metrics on explanations."""
    
    # Presentation dimension metrics
    PRESENTATION_METRICS = ['AFL', 'AC', 'CI']
    
    # Domain reasoning dimension metrics
    DOMAIN_METRICS = ['DTC', 'CEA', 'FS']
    
    def __init__(self, domain: str = 'general'):
        """
        Initialize SEF metrics.
        
        Args:
            domain: Domain for terminology matching ('legal', 'medical', 'financial')
        """
        self.domain = domain
        self.domain_terms = DOMAIN_TERMS.get(domain, [])
    
    def compute_all(
        self,
        explanation: str,
        predicted_answer: str,
        gold_answer: str,
        question: str = '',
    ) -> Dict[str, float]:
        """
        Compute all 6 SEF metrics.
        
        Returns:
            Dict with metric names as keys and scores (0-1) as values
        """
        return {
            'AFL': self.compute_afl(explanation, predicted_answer),
            'AC': self.compute_ac(explanation, predicted_answer),
            'CI': self.compute_ci(explanation),
            'DTC': self.compute_dtc(explanation),
            'CEA': self.compute_cea(explanation, predicted_answer),
            'FS': self.compute_fs(explanation),
        }
    
    def compute_afl(self, explanation: str, predicted_answer: str) -> float:
        """
        AFL: Answer First/Last
        
        Measures whether the answer is stated at the beginning AND/OR end of the explanation.
        Score: 1.0 if both, 0.5 if either, 0.0 if neither
        """
        if not explanation or not predicted_answer:
            return 0.0
        
        explanation_lower = explanation.lower()
        answer_lower = predicted_answer.lower()
        
        # Check first 200 characters
        first_part = explanation_lower[:200]
        has_first = self._contains_answer(first_part, answer_lower)
        
        # Check last 200 characters
        last_part = explanation_lower[-200:]
        has_last = self._contains_answer(last_part, answer_lower)
        
        if has_first and has_last:
            return 1.0
        elif has_first or has_last:
            return 0.5
        return 0.0
    
    def compute_ac(self, explanation: str, predicted_answer: str) -> float:
        """
        AC: Answer Clarity
        
        Measures how clearly and unambiguously the answer is stated.
        Looks for explicit answer patterns like "My answer is:", "Final Answer:", etc.
        """
        if not explanation:
            return 0.0
        
        explanation_lower = explanation.lower()
        
        # Strong patterns (clear, unambiguous)
        strong_patterns = [
            r'my answer is[:\s]+\*?\*?(?:\d+[.\)]\s*)?(yes|no)',
            r'final answer[:\s]+\*?\*?(?:\d+[.\)]\s*)?(yes|no)',
            r'the (?:correct )?answer is[:\s]+\*?\*?(?:\d+[.\)]\s*)?(yes|no)',
            r'\*\*answer[:\s]*\*?\*?\s*(?:\d+[.\)]\s*)?(yes|no)',
            r'conclusion[:\s]+.*?(yes|no)\s*\.?$',
        ]
        
        # Medium patterns (somewhat clear)
        medium_patterns = [
            r'therefore[,:\s]+(yes|no)',
            r'thus[,:\s]+(yes|no)',
            r'in conclusion[,:\s]+(yes|no)',
        ]
        
        # Check strong patterns
        for pattern in strong_patterns:
            if re.search(pattern, explanation_lower, re.MULTILINE | re.IGNORECASE):
                return 1.0
        
        # Check medium patterns
        for pattern in medium_patterns:
            if re.search(pattern, explanation_lower, re.MULTILINE | re.IGNORECASE):
                return 0.7
        
        # Weak: just contains the answer somewhere
        if predicted_answer and predicted_answer.lower() in explanation_lower:
            return 0.3
        
        return 0.0
    
    def compute_ci(self, explanation: str) -> float:
        """
        CI: Conclusion Isolation
        
        Measures whether the conclusion is separated from the reasoning body.
        Looks for conclusion markers and structural separation.
        """
        if not explanation:
            return 0.0
        
        explanation_lower = explanation.lower()
        
        # Strong conclusion markers (paper: "explicit conclusion header")
        strong_markers = [
            r'\*\*conclusion\*?\*?[:\s]',
            r'\n#+\s*conclusion',
            r'\nconclusion[:\s]',
        ]

        # Medium markers (paper: "in conclusion, to conclude, in summary, therefore, thus")
        medium_markers = [
            r'in conclusion',
            r'to conclude',
            r'therefore',
            r'thus',
            r'in summary',
        ]
        
        # Check strong markers
        for pattern in strong_markers:
            if re.search(pattern, explanation_lower):
                return 1.0
        
        # Check medium markers
        for pattern in medium_markers:
            if re.search(pattern, explanation_lower):
                return 0.6
        
        return 0.0
    
    def compute_dtc(self, explanation: str) -> float:
        """
        DTC: Domain Terminology Consistency
        
        Measures the use of domain-specific terminology.
        Score based on ratio of domain terms found.
        """
        if not explanation or not self.domain_terms:
            return 0.5  # Neutral if no domain terms defined
        
        explanation_lower = explanation.lower()
        
        # Count domain terms found
        terms_found = 0
        for term in self.domain_terms:
            if term.lower() in explanation_lower:
                terms_found += 1
        
        # Calculate score (expect at least 3-5 terms for a good explanation)
        if terms_found >= 5:
            return 1.0
        elif terms_found >= 3:
            return 0.8
        elif terms_found >= 1:
            return 0.5
        return 0.2
    
    def compute_cea(self, explanation: str, predicted_answer: str) -> float:
        """
        CEA: Conclusion Evidence Alignment
        
        Measures how well the conclusion is grounded in evidence/facts.
        Looks for evidence-to-conclusion linking patterns.
        """
        if not explanation:
            return 0.0
        
        explanation_lower = explanation.lower()
        
        # Evidence linking patterns
        linking_patterns = [
            r'based on (the|this|these)',
            r'according to',
            r'as (stated|mentioned|shown|indicated)',
            r'this (shows|indicates|suggests|demonstrates)',
            r'therefore',
            r'supports? (the|this|my)',
            r'evidence (shows|suggests|indicates)',
            r'the fact that',
            r'given (that|the)',
        ]
        
        # Analysis/reasoning patterns
        analysis_patterns = [
            r'\*\*analysis\*?\*?[:\s]',
            r'\nanalysis[:\s]',
            r'analyzing',
            r'examining',
            r'considering',
        ]
        
        link_count = sum(1 for p in linking_patterns if re.search(p, explanation_lower))
        analysis_count = sum(1 for p in analysis_patterns if re.search(p, explanation_lower))
        
        # Score based on counts
        if link_count >= 3 and analysis_count >= 1:
            return 1.0
        elif link_count >= 2:
            return 0.8
        elif link_count >= 1:
            return 0.5
        elif analysis_count >= 1:
            return 0.3
        return 0.0
    
    def compute_fs(self, explanation: str) -> float:
        """
        FS: Fact Specificity
        
        Measures the use of specific facts vs vague statements.
        Looks for specific references, quotes, numbers.
        """
        if not explanation:
            return 0.0
        
        explanation_lower = explanation.lower()
        
        # Specificity indicators
        specific_patterns = [
            r'"[^"]+?"',  # Quoted text
            r'\d+',  # Numbers
            r'specifically',
            r'in particular',
            r'for example',
            r'such as',
            r'\*\*key facts?\*?\*?[:\s]',
            r'the fact that',
            r'fact \d+',
            r'first[,:\s]',
            r'second[,:\s]',
        ]
        
        # Vague indicators (negative)
        vague_patterns = [
            r'generally',
            r'usually',
            r'often',
            r'sometimes',
            r'may or may not',
            r'could be',
            r'might be',
            r'possibly',
        ]
        
        specific_count = sum(1 for p in specific_patterns if re.search(p, explanation_lower))
        vague_count = sum(1 for p in vague_patterns if re.search(p, explanation_lower))
        
        # Score based on balance
        if specific_count >= 4 and vague_count <= 1:
            return 1.0
        elif specific_count >= 3:
            return 0.8
        elif specific_count >= 2:
            return 0.6
        elif specific_count >= 1:
            return 0.4
        return 0.2
    
    def _contains_answer(self, text: str, answer: str) -> bool:
        """Check if text contains the answer."""
        if not text or not answer:
            return False
        
        # Direct match
        if answer in text:
            return True
        
        # Pattern match
        pattern = rf'\b{re.escape(answer)}\b'
        return bool(re.search(pattern, text, re.IGNORECASE))


def process_result_file(
    filepath: Path,
    metrics_calculator: SEFMetrics
) -> Optional[Dict[str, Any]]:
    """Process a single result file and compute metrics."""
    try:
        with open(filepath) as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load {filepath}: {e}")
        return None
    
    results = data.get('results', [])
    if not results:
        logger.warning(f"No results in {filepath}")
        return None
    
    # Update domain for metrics
    domain = data.get('domain', 'general')
    metrics_calculator.domain = domain
    metrics_calculator.domain_terms = DOMAIN_TERMS.get(domain, [])
    
    # Compute metrics for each sample
    sample_metrics = []
    for sample in results:
        explanation = sample.get('explanation', '')
        predicted_answer = sample.get('predicted_answer', '')
        gold_answer = sample.get('gold_answer', '')
        question = sample.get('question', '')
        
        if not explanation:
            continue
        
        scores = metrics_calculator.compute_all(
            explanation, predicted_answer, gold_answer, question
        )
        
        sample_metrics.append({
            'sample_id': sample.get('sample_id', ''),
            'is_correct': sample.get('is_correct', False),
            'metrics': scores
        })
    
    if not sample_metrics:
        return None
    
    # Aggregate metrics
    n_samples = len(sample_metrics)
    aggregated = {}
    for metric_name in ['AFL', 'AC', 'CI', 'DTC', 'CEA', 'FS']:
        aggregated[metric_name] = sum(
            s['metrics'][metric_name] for s in sample_metrics
        ) / n_samples
    
    return {
        'source_file': filepath.name,
        'model': data.get('model', ''),
        'method': data.get('method', ''),
        'domain': domain,
        'dataset': data.get('dataset', ''),
        'accuracy': data.get('accuracy', 0.0),
        'n_samples': n_samples,
        'aggregated_metrics': aggregated,
        'sample_metrics': sample_metrics
    }


def generate_metrics_report(
    all_results: List[Dict],
    output_dir: Path
) -> str:
    """Generate a comprehensive metrics report."""
    if not all_results:
        return "No results to report."
    
    lines = []
    lines.append("=" * 70)
    lines.append("SEF METRICS COMPUTATION REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Total experiments: {len(all_results)}")
    lines.append(f"Total samples: {sum(r['n_samples'] for r in all_results)}")
    lines.append("")
    
    # Global metric averages
    lines.append("-" * 70)
    lines.append("MEAN METRIC SCORES (Global)")
    lines.append("-" * 70)
    lines.append("")
    
    total_samples = sum(r['n_samples'] for r in all_results)
    global_avg = {m: 0.0 for m in ['AFL', 'AC', 'CI', 'DTC', 'CEA', 'FS']}
    
    for result in all_results:
        weight = result['n_samples'] / total_samples
        for m in global_avg:
            global_avg[m] += result['aggregated_metrics'][m] * weight
    
    lines.append("Presentation Dimension:")
    for m in ['AFL', 'AC', 'CI']:
        lines.append(f"  {m}: {global_avg[m]:.4f}")
    
    lines.append("")
    lines.append("Domain Reasoning Dimension:")
    for m in ['DTC', 'CEA', 'FS']:
        lines.append(f"  {m}: {global_avg[m]:.4f}")
    
    # Metrics by method
    lines.append("")
    lines.append("-" * 70)
    lines.append("MEAN METRIC SCORES BY METHOD")
    lines.append("-" * 70)
    lines.append("")
    
    method_metrics = defaultdict(lambda: defaultdict(list))
    for result in all_results:
        method = result['method']
        for m in global_avg:
            method_metrics[method][m].append(result['aggregated_metrics'][m])
    
    header = f"{'Method':<25}"
    for m in ['AFL', 'AC', 'CI', 'DTC', 'CEA', 'FS']:
        header += f" {m:>6}"
    lines.append(header)
    lines.append("-" * 70)
    
    for method in sorted(method_metrics.keys()):
        row = f"{method:<25}"
        for m in ['AFL', 'AC', 'CI', 'DTC', 'CEA', 'FS']:
            vals = method_metrics[method][m]
            avg = sum(vals) / len(vals) if vals else 0
            row += f" {avg:>6.3f}"
        lines.append(row)
    
    # Metrics by domain
    lines.append("")
    lines.append("-" * 70)
    lines.append("MEAN METRIC SCORES BY DOMAIN")
    lines.append("-" * 70)
    lines.append("")
    
    domain_metrics = defaultdict(lambda: defaultdict(list))
    for result in all_results:
        domain = result['domain']
        for m in global_avg:
            domain_metrics[domain][m].append(result['aggregated_metrics'][m])
    
    header = f"{'Domain':<15}"
    for m in ['AFL', 'AC', 'CI', 'DTC', 'CEA', 'FS']:
        header += f" {m:>6}"
    lines.append(header)
    lines.append("-" * 50)
    
    for domain in sorted(domain_metrics.keys()):
        row = f"{domain:<15}"
        for m in ['AFL', 'AC', 'CI', 'DTC', 'CEA', 'FS']:
            vals = domain_metrics[domain][m]
            avg = sum(vals) / len(vals) if vals else 0
            row += f" {avg:>6.3f}"
        lines.append(row)
    
    lines.append("")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Compute SEF metrics on experiment explanations'
    )
    parser.add_argument(
        '--results-dir', type=str, default='results/raw',
        help='Directory containing raw result files'
    )
    parser.add_argument(
        '--output-dir', type=str, default='results/metrics',
        help='Output directory for metric scores'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress stdout output'
    )
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    results_dir = project_root / args.results_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return
    
    # Initialize metrics calculator
    metrics_calculator = SEFMetrics()
    
    # Process all result files
    result_files = list(results_dir.glob('*.json'))
    logger.info(f"Found {len(result_files)} result files")
    
    all_results = []
    for filepath in result_files:
        logger.info(f"Processing: {filepath.name}")
        result = process_result_file(filepath, metrics_calculator)
        if result:
            all_results.append(result)
            
            # Save individual metric file
            output_file = output_dir / f"metrics_{filepath.stem}.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
    
    logger.info(f"Processed {len(all_results)} files with explanations")
    
    # Generate report
    report = generate_metrics_report(all_results, output_dir)
    
    if not args.quiet:
        print(report)
    
    # Save report
    report_file = output_dir / "sef_metrics_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    logger.info(f"Report saved to {report_file}")
    
    # Save aggregated results
    aggregated_file = output_dir / "aggregated_metrics.json"
    aggregated_data = {
        'total_experiments': len(all_results),
        'total_samples': sum(r['n_samples'] for r in all_results),
        'experiments': [
            {
                'model': r['model'],
                'method': r['method'],
                'domain': r['domain'],
                'dataset': r['dataset'],
                'accuracy': r['accuracy'],
                'n_samples': r['n_samples'],
                'metrics': r['aggregated_metrics']
            }
            for r in all_results
        ]
    }
    with open(aggregated_file, 'w') as f:
        json.dump(aggregated_data, f, indent=2)
    logger.info(f"Aggregated metrics saved to {aggregated_file}")


if __name__ == '__main__':
    main()

