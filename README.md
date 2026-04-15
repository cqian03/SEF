# SEF: Structured Explainability Framework

Code for the paper:

> **From "Thinking" to "Justifying": Aligning High-Stakes Explainability with Professional Communication Standards**
>
> ACL 2026 Findings

## Overview

SEF operationalizes professional communication conventions (CREAC, BLUF) through six metrics for structure and grounding. The "Result → Justify" paradigm states a conclusion first, then builds a structured defense — unlike Chain-of-Thought which reasons before concluding.

## Repository Structure

```
├── src/
│   ├── data_loader.py              # Multi-domain dataset loader
│   ├── baselines/
│   │   ├── direct_prompting.py     # Direct (no explanation)
│   │   ├── standard_cot.py         # Chain-of-Thought
│   │   ├── tree_of_thought.py      # Tree-of-Thought
│   │   ├── chain_of_verification.py# Chain-of-Verification
│   │   ├── vanilla_rag.py          # Vanilla RAG
│   │   ├── self_rag.py             # Self-RAG
│   │   └── sef_prompting.py        # SEF (our method) + ablations
│   ├── llm_clients/
│   │   └── vllm_client.py          # vLLM inference client
│   └── utils/
│       └── answer_extractor.py     # Yes/No answer extraction
├── scripts/
│   ├── download_data.py            # Download datasets
│   ├── start_vllm_server.sh        # Start vLLM server
│   ├── run_baselines.py            # Run baseline experiments
│   ├── run_sef_experiments.py      # Run SEF + ablations
│   ├── analyze_metrics.py          # Compute 6 SEF metrics
│   └── analyze_correlation.py      # Metric-accuracy correlations
├── configs/
│   └── sef_config.yaml             # Experiment configuration
└── requirements.txt
```

## Reproducing Experiments

```bash
# Install dependencies
pip install -r requirements.txt

# Download datasets
python scripts/download_data.py

# Start vLLM server (one model at a time)
bash scripts/start_vllm_server.sh deepseek-14b

# Run baselines
python scripts/run_baselines.py --model deepseek_r1_14b

# Run SEF + ablations
python scripts/run_sef_experiments.py --model deepseek_r1_14b

# Compute metrics and correlations
python scripts/analyze_metrics.py --results-dir results/raw
python scripts/analyze_correlation.py --metrics-dir results/metrics
```

Repeat for each model: `gemma-12b`, `ministral-14b`, `qwen-14b`.

## Models (Section 4.1, Appendix A.5)

| Paper Name | HuggingFace ID | Params |
|---|---|---|
| DeepSeek-R1-Distill-Qwen-14B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | 14B |
| Gemma 3 12B | `google/gemma-3-12b-it` | 12B |
| Ministral-3-14B | `mistralai/Ministral-3-14B-Instruct-2512` | 14B |
| Qwen 2.5 14B | `Qwen/Qwen2.5-14B-Instruct` | 14B |

All experiments use greedy decoding (temperature=0) via vLLM.

## Datasets (Section 4.1, Appendix A.4)

| Task | Domain | Source |
|---|---|---|
| FPB | Financial | `nickmuchi/financial-classification` |
| ConsumerQA | Legal | `nguha/legalbench` (consumer_contracts_qa) |
| Hearsay | Legal | `nguha/legalbench` (hearsay) |
| PubMedQA | Medical | `qiaojin/PubMedQA` (pqa_labeled, Yes/No only) |

## SEF Metrics (Section 3)

| Metric | Dimension | CREAC Element | BLUF Principle |
|---|---|---|---|
| AFL (Answer First/Last) | Plausibility | Bookend C–C | Lead with the answer |
| AC (Answer Clarity) | Plausibility | Explicit C | Unambiguous bottom line |
| CI (Conclusion Isolation) | Plausibility | Separated C | Distinct summary block |
| DTC (Domain Terminology) | Faithfulness | R/E precision | Domain terminology |
| CEA (Conclusion-Evidence Alignment) | Faithfulness | A links evidence | Evidence-linked defense |
| FS (Fact Specificity) | Faithfulness | R/E specifics | Concrete over vague |

C=Conclusion, R=Rule, E=Explanation, A=Analysis (CREAC components).

## Citation

```bibtex
@misc{qian2026thinkingjustifyingaligninghighstakes,
      title={From "Thinking" to "Justifying": Aligning High-Stakes Explainability with Professional Communication Standards}, 
      author={Chen Qian and Yimeng Wang and Yu Chen and Lingfei Wu and Andreas Stathopoulos},
      year={2026},
      eprint={2601.07233},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2601.07233}, 
}
```

## License

This repository uses publicly released research benchmarks and open-weight models under their respective licenses.
