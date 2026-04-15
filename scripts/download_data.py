#!/usr/bin/env python3
"""
Download all datasets for SEF experiments.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --domain legal
    python scripts/download_data.py --domain medical
    python scripts/download_data.py --domain financial
"""

import os
import sys
import argparse
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_legal_datasets(cache_dir: str):
    """Download LegalBench datasets."""
    from huggingface_hub import hf_hub_download
    
    logger.info("Downloading Legal domain datasets...")
    
    datasets = [
        'hearsay',
        'consumer_contracts_qa',
    ]
    
    for dataset in datasets:
        try:
            logger.info(f"  Downloading {dataset}...")
            file_path = f"data/{dataset}/test.tsv"
            local_path = hf_hub_download(
                repo_id="nguha/legalbench",
                filename=file_path,
                repo_type="dataset",
                cache_dir=os.path.join(cache_dir, 'legalbench'),
            )
            logger.info(f"    ✓ {dataset} -> {local_path}")
        except Exception as e:
            logger.warning(f"    ✗ {dataset}: {e}")
    
    logger.info("Legal datasets download complete!")


def download_medical_datasets(cache_dir: str):
    """Download PubMedQA dataset."""
    from datasets import load_dataset
    
    logger.info("Downloading Medical domain datasets...")
    
    try:
        logger.info("  Downloading PubMedQA...")
        dataset = load_dataset(
            "qiaojin/PubMedQA",
            "pqa_labeled",
            cache_dir=os.path.join(cache_dir, 'pubmedqa'),
        )
        logger.info(f"    ✓ PubMedQA: {len(dataset['train'])} samples")
    except Exception as e:
        logger.warning(f"    ✗ PubMedQA: {e}")
    
    logger.info("Medical datasets download complete!")


def download_financial_datasets(cache_dir: str):
    """Download Financial PhraseBank dataset."""
    from datasets import load_dataset
    
    logger.info("Downloading Financial domain datasets...")
    
    try:
        logger.info("  Downloading Financial PhraseBank...")
        dataset = load_dataset(
            "nickmuchi/financial-classification",
            cache_dir=os.path.join(cache_dir, 'fpb'),
        )
        split = dataset.get('test', dataset.get('train', list(dataset.values())[0]))
        logger.info(f"    ✓ FPB: {len(split)} samples")
    except Exception as e:
        logger.warning(f"    ✗ FPB: {e}")
    
    logger.info("Financial datasets download complete!")


def verify_downloads(cache_dir: str):
    """Verify all datasets are downloaded and loadable."""
    from src.data_loader import MultiDomainLoader
    
    logger.info("\n" + "=" * 60)
    logger.info("VERIFYING DOWNLOADS")
    logger.info("=" * 60)
    
    loader = MultiDomainLoader(cache_dir=cache_dir)
    
    all_ok = True
    for domain in ['legal', 'medical', 'financial']:
        datasets = loader.get_available_datasets(domain)[domain]
        for dataset in datasets:
            try:
                samples = loader.load(domain, dataset, max_samples=5)
                status = f"✓ {len(samples)} samples"
            except Exception as e:
                status = f"✗ Error: {e}"
                all_ok = False
            logger.info(f"  {domain}/{dataset}: {status}")
    
    if all_ok:
        logger.info("\n✓ All datasets verified successfully!")
    else:
        logger.warning("\n⚠ Some datasets failed verification!")
    
    return all_ok


def main():
    parser = argparse.ArgumentParser(description='Download SEF datasets')
    parser.add_argument('--domain', type=str, choices=['legal', 'medical', 'financial', 'all'],
                       default='all', help='Domain to download')
    parser.add_argument('--cache-dir', type=str, default='data',
                       help='Cache directory for datasets')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify existing downloads')
    
    args = parser.parse_args()
    
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    
    if args.verify_only:
        verify_downloads(cache_dir)
        return
    
    logger.info("=" * 60)
    logger.info("SEF DATASET DOWNLOAD")
    logger.info("=" * 60)
    logger.info(f"Cache directory: {cache_dir}")
    
    if args.domain in ['legal', 'all']:
        download_legal_datasets(cache_dir)
    
    if args.domain in ['medical', 'all']:
        download_medical_datasets(cache_dir)
    
    if args.domain in ['financial', 'all']:
        download_financial_datasets(cache_dir)
    
    # Verify downloads
    verify_downloads(cache_dir)


if __name__ == "__main__":
    main()

