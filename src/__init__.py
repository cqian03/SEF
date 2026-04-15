"""
SEF: Structured Explainability Framework

A prompting framework that operationalizes professional communication conventions
(CREAC, BLUF) for structured, verifiable AI explanations in high-stakes domains.

Paper: "From 'Thinking' to 'Justifying': Aligning High-Stakes Explainability
       with Professional Communication Standards" (ACL 2026 Findings)
"""

from .data_loader import MultiDomainLoader, Sample

__all__ = [
    'MultiDomainLoader',
    'Sample',
]
