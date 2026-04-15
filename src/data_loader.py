"""
Multi-Domain Data Loader for SEF (Structured Explainability Framework)

Supports Yes/No binary classification datasets across:
- Legal: hearsay, consumer_contracts_qa (from LegalBench)
- Medical: PubMedQA (pqa_labeled, Yes/No only)
- Financial: Financial PhraseBank (positive/negative only)

Paper Section 4.1 & Appendix A.4:
  1,618 test samples total across 4 tasks in 3 domains.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

logger = logging.getLogger(__name__)


@dataclass
class Sample:
    """A single sample from any domain benchmark."""
    id: str
    question: str
    context: str
    choices: List[str]
    answer: str
    domain: str
    dataset_name: str
    metadata: Dict[str, Any]


class MultiDomainLoader:
    """
    Unified loader for Yes/No datasets across Legal, Medical, and Financial domains.
    """

    DATASETS = {
        'legal': {
            'hearsay': {
                'hf_id': 'nguha/legalbench',
                'subset': 'hearsay',
                'format': 'yes_no'
            },
            'consumer_contracts_qa': {
                'hf_id': 'nguha/legalbench',
                'subset': 'consumer_contracts_qa',
                'format': 'yes_no'
            },
        },
        'medical': {
            'pubmedqa': {
                'hf_id': 'qiaojin/PubMedQA',
                'subset': 'pqa_labeled',
                'format': 'yes_no_maybe'
            }
        },
        'financial': {
            'fpb': {
                'hf_id': 'nickmuchi/financial-classification',
                'subset': None,
                'format': 'sentiment'
            }
        }
    }

    def __init__(self, cache_dir: str = "data"):
        if load_dataset is None:
            raise ImportError("datasets package not installed. Run: pip install datasets")
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Initialized MultiDomainLoader with cache: {cache_dir}")

    def get_available_datasets(self, domain: Optional[str] = None) -> Dict[str, List[str]]:
        """Get available datasets, optionally filtered by domain."""
        if domain:
            return {domain: list(self.DATASETS.get(domain, {}).keys())}
        return {d: list(datasets.keys()) for d, datasets in self.DATASETS.items()}

    def load(
        self,
        domain: str,
        dataset_name: str,
        split: str = "test",
        max_samples: Optional[int] = None
    ) -> List[Sample]:
        """Load a specific dataset from a domain."""
        if domain not in self.DATASETS:
            raise ValueError(f"Unknown domain: {domain}. Available: {list(self.DATASETS.keys())}")
        if dataset_name not in self.DATASETS[domain]:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available for {domain}: {list(self.DATASETS[domain].keys())}")

        config = self.DATASETS[domain][dataset_name]
        logger.info(f"Loading {domain}/{dataset_name} ({split} split)")

        if domain == 'legal':
            samples = self._load_legalbench(dataset_name, config, split)
        elif domain == 'medical':
            samples = self._load_pubmedqa(config, split)
        elif domain == 'financial':
            samples = self._load_fpb(config, split)
        else:
            samples = []

        if max_samples and len(samples) > max_samples:
            samples = samples[:max_samples]

        logger.info(f"Loaded {len(samples)} samples from {domain}/{dataset_name}")
        return samples

    def load_domain(self, domain: str, split: str = "test", max_samples: Optional[int] = None) -> Dict[str, List[Sample]]:
        """Load all datasets for a domain."""
        if domain not in self.DATASETS:
            raise ValueError(f"Unknown domain: {domain}")
        all_samples = {}
        for dataset_name in self.DATASETS[domain]:
            all_samples[dataset_name] = self.load(domain, dataset_name, split, max_samples)
        return all_samples

    def load_all(self, split: str = "test", max_samples: Optional[int] = None) -> Dict[str, Dict[str, List[Sample]]]:
        """Load all datasets from all domains."""
        all_data = {}
        for domain in self.DATASETS:
            all_data[domain] = self.load_domain(domain, split, max_samples)
        return all_data

    # ==================== Legal Domain Loaders ====================

    def _load_legalbench(self, dataset_name: str, config: Dict, split: str) -> List[Sample]:
        """Load LegalBench datasets."""
        try:
            dataset = load_dataset(
                config['hf_id'],
                config['subset'],
                cache_dir=os.path.join(self.cache_dir, 'legalbench'),
            )
            data_split = dataset.get(split, dataset.get('train', list(dataset.values())[0]))
            samples = []
            for idx, item in enumerate(data_split):
                sample = self._parse_legalbench_item(item, dataset_name, idx)
                samples.append(sample)
            return samples
        except Exception as e:
            logger.debug(f"HF dataset API unavailable for {dataset_name}: {e}")
            return self._try_load_legalbench_tsv(dataset_name, config, split)

    def _try_load_legalbench_tsv(self, dataset_name: str, config: Dict, split: str) -> List[Sample]:
        """Try loading LegalBench from TSV files (fallback mechanism)."""
        if hf_hub_download is None:
            logger.warning(f"Cannot load {dataset_name}: huggingface_hub not installed")
            return []
        try:
            import pandas as pd
            file_path = f"data/{config['subset']}/{split}.tsv"
            local_path = hf_hub_download(
                repo_id=config['hf_id'],
                filename=file_path,
                repo_type="dataset",
                cache_dir=os.path.join(self.cache_dir, 'legalbench'),
            )
            df = pd.read_csv(local_path, sep='\t')
            samples = []
            for idx, row in df.iterrows():
                item = row.to_dict()
                sample = self._parse_legalbench_item(item, dataset_name, idx)
                samples.append(sample)
            return samples
        except Exception as e:
            logger.warning(f"TSV loading failed for {dataset_name}: {e}")
            return []

    def _parse_legalbench_item(self, item: Dict, dataset_name: str, idx: int) -> Sample:
        """Parse a LegalBench item into a Sample."""
        if dataset_name == 'hearsay':
            return Sample(
                id=f"legal_hearsay_{idx}",
                question=item.get('text', item.get('question', '')),
                context=item.get('context', ''),
                choices=['Yes', 'No'],
                answer=item.get('answer', item.get('label', '')),
                domain='legal',
                dataset_name='hearsay',
                metadata={'original': item}
            )
        elif dataset_name == 'consumer_contracts_qa':
            return Sample(
                id=f"legal_consumer_{idx}",
                question=item.get('question', item.get('text', '')),
                context=item.get('contract', item.get('context', '')),
                choices=['Yes', 'No'],
                answer=item.get('answer', item.get('label', '')),
                domain='legal',
                dataset_name='consumer_contracts_qa',
                metadata={'original': item}
            )
        else:
            return Sample(
                id=f"legal_{dataset_name}_{idx}",
                question=item.get('question', item.get('text', '')),
                context=item.get('context', ''),
                choices=['Yes', 'No'],
                answer=item.get('answer', item.get('label', '')),
                domain='legal',
                dataset_name=dataset_name,
                metadata={'original': item}
            )

    # ==================== Medical Domain Loaders ====================

    def _load_pubmedqa(self, config: Dict, split: str) -> List[Sample]:
        """Load PubMedQA dataset (filtering to Yes/No only)."""
        try:
            dataset = load_dataset(
                config['hf_id'],
                config['subset'],
                cache_dir=os.path.join(self.cache_dir, 'pubmedqa'),
            )
            data_split = dataset.get(split, dataset.get('train', list(dataset.values())[0]))
            samples = []
            for idx, item in enumerate(data_split):
                answer = item.get('final_decision', item.get('label', ''))
                if str(answer).lower() in ['yes', 'no']:
                    sample = Sample(
                        id=f"medical_pubmedqa_{idx}",
                        question=item.get('question', ''),
                        context=self._format_pubmedqa_context(item),
                        choices=['Yes', 'No'],
                        answer=str(answer).capitalize(),
                        domain='medical',
                        dataset_name='pubmedqa',
                        metadata={'original': item}
                    )
                    samples.append(sample)
            return samples
        except Exception as e:
            logger.warning(f"Failed to load PubMedQA: {e}")
            return []

    def _format_pubmedqa_context(self, item: Dict) -> str:
        """Format PubMedQA context from abstracts."""
        contexts = item.get('context', {})
        if isinstance(contexts, dict):
            ctx_list = contexts.get('contexts', [])
            if isinstance(ctx_list, list):
                return ' '.join(ctx_list)
        elif isinstance(contexts, str):
            return contexts
        return item.get('long_answer', item.get('abstract', ''))

    # ==================== Financial Domain Loaders ====================

    def _load_fpb(self, config: Dict, split: str) -> List[Sample]:
        """Load Financial sentiment dataset.

        Converts sentiment (positive/negative/neutral) to Yes/No:
        - Question: "Is the sentiment of this financial statement positive?"
        - positive -> Yes, negative -> No, neutral -> filtered out
        """
        try:
            if config['subset']:
                dataset = load_dataset(
                    config['hf_id'], config['subset'],
                    cache_dir=os.path.join(self.cache_dir, 'fpb'),
                )
            else:
                dataset = load_dataset(
                    config['hf_id'],
                    cache_dir=os.path.join(self.cache_dir, 'fpb'),
                )
            data_split = dataset.get(split, dataset.get('test', dataset.get('train', list(dataset.values())[0])))
            samples = []
            for idx, item in enumerate(data_split):
                label = item.get('labels', item.get('label', 1))
                sentence = item.get('text', item.get('sentence', ''))
                if label == 1:  # Skip neutral
                    continue
                answer = 'Yes' if label == 2 else 'No'
                sample = Sample(
                    id=f"financial_fpb_{idx}",
                    question="Is the sentiment of this financial statement positive?",
                    context=sentence,
                    choices=['Yes', 'No'],
                    answer=answer,
                    domain='financial',
                    dataset_name='fpb',
                    metadata={'original': item, 'original_label': label}
                )
                samples.append(sample)
            return samples
        except Exception as e:
            logger.warning(f"Failed to load FPB: {e}")
            return []


def load_dataset_samples(
    domain: str,
    dataset_name: str,
    split: str = "test",
    max_samples: Optional[int] = None,
    cache_dir: str = "data"
) -> List[Sample]:
    """Convenience function to load samples from a specific dataset."""
    loader = MultiDomainLoader(cache_dir=cache_dir)
    return loader.load(domain, dataset_name, split, max_samples)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = MultiDomainLoader()
    for domain in loader.DATASETS:
        for ds_name in loader.DATASETS[domain]:
            try:
                samples = loader.load(domain, ds_name, max_samples=5)
                print(f"  {domain}/{ds_name}: {len(samples)} samples loaded")
            except Exception as e:
                print(f"  {domain}/{ds_name}: FAILED - {e}")
