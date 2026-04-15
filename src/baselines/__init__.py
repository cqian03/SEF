"""
Baseline and SEF implementations used in the paper.

Methods (Table 2 in the paper):
- DirectPrompting: Direct question answering without reasoning
- StandardCoT: Chain-of-Thought prompting
- TreeOfThought: Tree-of-Thought deliberate reasoning
- ChainOfVerification: Self-verification prompting
- VanillaRAG: Basic retrieve-then-generate
- SelfRAG: Retrieval with self-reflection
- SEFPrompting: Structured Explainability Framework (our method)
"""

from .direct_prompting import DirectPrompting
from .standard_cot import StandardCoT
from .tree_of_thought import TreeOfThought
from .chain_of_verification import ChainOfVerification
from .vanilla_rag import VanillaRAG
from .self_rag import SelfRAG
from .sef_prompting import SEFPrompting, SEFAblation

__all__ = [
    "DirectPrompting",
    "StandardCoT",
    "TreeOfThought",
    "ChainOfVerification",
    "VanillaRAG",
    "SelfRAG",
    "SEFPrompting",
    "SEFAblation",
]


def get_baseline(name: str, **kwargs):
    """Factory function to get the appropriate baseline.

    Args:
        name: Baseline name. Options:
            - "direct": Direct prompting
            - "standard_cot": Chain-of-Thought
            - "tree_of_thought": Tree-of-Thought
            - "chain_of_verification": Chain-of-Verification
            - "vanilla_rag": Vanilla RAG
            - "self_rag": Self-RAG
            - "sef": SEF prompting (our method)
        **kwargs: Additional configuration for the baseline

    Returns:
        Initialized baseline instance
    """
    baselines = {
        "direct": DirectPrompting,
        "standard_cot": StandardCoT,
        "tree_of_thought": TreeOfThought,
        "chain_of_verification": ChainOfVerification,
        "vanilla_rag": VanillaRAG,
        "self_rag": SelfRAG,
        "sef": SEFPrompting,
    }
    if name not in baselines:
        raise ValueError(f"Unknown baseline: {name}. Choose from {list(baselines.keys())}")
    return baselines[name](**kwargs)
