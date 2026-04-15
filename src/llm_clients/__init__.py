"""
LLM Client implementations for SEF.

Models used in the paper (Section 4.1):
- deepseek_r1_14b: DeepSeek-R1-Distill-Qwen-14B
- gemma_3_12b: Google Gemma 3 12B IT
- ministral_14b: Mistral Ministral-3-14B-Instruct-2512
- qwen_2_5_14b: Qwen 2.5 14B Instruct
"""

from .vllm_client import (
    VLLMClient,
    DeepSeekR1_14BClient,
    Gemma3_12BClient,
    Ministral14BClient,
    Qwen25_14BClient,
)

__all__ = [
    "VLLMClient",
    "DeepSeekR1_14BClient",
    "Gemma3_12BClient",
    "Ministral14BClient",
    "Qwen25_14BClient",
]


def get_client(provider: str, **kwargs):
    """
    Factory function to get the appropriate LLM client.

    Args:
        provider: Model identifier. Options:
            - "deepseek_r1_14b": DeepSeek R1 Distill Qwen 14B
            - "gemma_3_12b": Google Gemma 3 12B
            - "ministral_14b": Mistral Ministral 3 14B
            - "qwen_2_5_14b": Qwen 2.5 14B
        **kwargs: Additional configuration for the client

    Returns:
        Initialized LLM client
    """
    clients = {
        "deepseek_r1_14b": DeepSeekR1_14BClient,
        "gemma_3_12b": Gemma3_12BClient,
        "ministral_14b": Ministral14BClient,
        "qwen_2_5_14b": Qwen25_14BClient,
        "vllm": VLLMClient,
    }

    provider_lower = provider.lower()
    if provider_lower not in clients:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Choose from {list(clients.keys())}"
        )
    return clients[provider_lower](**kwargs)
