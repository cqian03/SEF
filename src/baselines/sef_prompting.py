"""
SEF: Structured Explanation Framework

A domain-agnostic prompting method designed based on empirically validated metrics
that correlate with reasoning accuracy across high-stakes domains.

Core Principles (6 validated metrics):
1. AFL - Answer First/Last: State answer at beginning AND end
2. AC  - Answer Clarity: Clear, unambiguous answer statement
3. CI  - Conclusion Isolation: Separate conclusion from reasoning
4. DTC - Domain Terminology Consistency: Use precise domain-specific terms
5. CEA - Conclusion Evidence Alignment: Ground conclusion in evidence
6. FS  - Fact Specificity: Use specific facts, not vague statements

Design Philosophy:
- Combines presentation quality (how answer is presented)
- With domain reasoning quality (how reasoning is structured)
- Works across Legal, Medical, and Financial domains
"""

import logging
from typing import Dict, Any, List, Optional

from src.utils.answer_extractor import extract_answer

logger = logging.getLogger(__name__)


# Domain-specific terminology instructions
DOMAIN_TERMINOLOGY = {
    'legal': "Use precise legal terminology consistently (e.g., 'hearsay', 'testimony', 'evidence', 'statute', 'precedent').",
    'medical': "Use precise medical terminology consistently (e.g., 'diagnosis', 'prognosis', 'symptoms', 'treatment', 'efficacy').",
    'financial': "Use precise financial terminology consistently (e.g., 'revenue', 'earnings', 'dividend', 'valuation', 'sentiment').",
    'general': "Use precise terminology consistently throughout your analysis."
}

DOMAIN_EXPERT_ROLE = {
    'legal': "You are a legal reasoning expert.",
    'medical': "You are a medical reasoning expert.",
    'financial': "You are a financial analysis expert.",
    'general': "You are a domain expert."
}


class SEFPrompting:
    """
    SEF: Structured Explanation Framework
    
    Prompting method designed based on 6 empirically validated metrics:
    - 3 Presentation metrics: AFL, AC, CI
    - 3 Domain reasoning metrics: DTC, CEA, FS
    """
    
    def __init__(
        self, 
        llm_client, 
        domain: str = 'general',
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize SEF prompting.
        
        Args:
            llm_client: LLM client
            domain: Domain for terminology ('legal', 'medical', 'financial', 'general')
            config: Optional configuration dict
        """
        self.llm_client = llm_client
        self.name = "sef"
        self.domain = domain
        self.config = config or {}
        
        # SEF components (can be toggled for ablation)
        self.use_answer_preview = self.config.get('answer_preview', True)  # AFL
        self.use_answer_clarity = self.config.get('answer_clarity', True)  # AC
        self.use_conclusion_isolation = self.config.get('conclusion_isolation', True)  # CI
        self.use_domain_terminology = self.config.get('domain_terminology', True)  # DTC
        self.use_evidence_alignment = self.config.get('evidence_alignment', True)  # CEA
        self.use_fact_specificity = self.config.get('fact_specificity', True)  # FS
        
        logger.info(f"Initialized SEFPrompting for domain={domain} with config: {self.config}")
    
    def set_domain(self, domain: str):
        """Update the domain for terminology."""
        self.domain = domain
        logger.info(f"SEF domain set to: {domain}")
    
    def _build_prompt(
        self,
        context: str,
        question: str,
        choices: Optional[List[str]] = None,
    ) -> str:
        """Build the SEF prompt with all 6 components."""
        
        prompt_parts = []
        
        # System instruction with domain-specific role
        role = DOMAIN_EXPERT_ROLE.get(self.domain, DOMAIN_EXPERT_ROLE['general'])
        prompt_parts.append(f"{role} Analyze the following question with precision and clarity.")
        
        # Domain terminology instruction (DTC)
        if self.use_domain_terminology:
            terminology = DOMAIN_TERMINOLOGY.get(self.domain, DOMAIN_TERMINOLOGY['general'])
            prompt_parts.append(terminology)
        
        # Fact specificity instruction (FS)
        if self.use_fact_specificity:
            prompt_parts.append("Cite specific facts from the context, not vague generalizations.")
        
        prompt_parts.append("")
        
        # Context and question
        prompt_parts.append(f"**Context:**\n{context}\n")
        prompt_parts.append(f"**Question:**\n{question}\n")
        
        # For Yes/No tasks (our focus)
        is_binary = choices is None or len(choices) <= 2
        
        # Instructions based on enabled components
        prompt_parts.append("**Please structure your analysis as follows:**\n")
        
        # Component 1: Answer Preview (AFL - Answer First)
        if self.use_answer_preview:
            prompt_parts.append("**ANSWER PREVIEW:** State your answer upfront (Yes or No).")
        
        # Component 2: Fact Specificity (FS)
        if self.use_fact_specificity:
            prompt_parts.append("\n**KEY FACTS:** List 2-3 specific facts from the context that are most relevant.")
        
        # Component 3: Evidence Alignment (CEA) + Domain Terminology (DTC)
        if self.use_evidence_alignment or self.use_domain_terminology:
            prompt_parts.append("\n**ANALYSIS:**")
            if self.use_domain_terminology:
                prompt_parts.append("- Use precise domain terminology")
            if self.use_evidence_alignment:
                prompt_parts.append("- Explain how each fact supports your answer")
        
        # Component 4: Conclusion Isolation (CI) + Answer Clarity (AC) + AFL (Answer Last)
        if self.use_conclusion_isolation:
            prompt_parts.append("\n**CONCLUSION:**")
            if self.use_answer_clarity:
                prompt_parts.append("- State your final answer clearly and unambiguously")
            if self.use_evidence_alignment:
                prompt_parts.append("- Summarize the key evidence supporting your answer")
            prompt_parts.append("- End with: **My answer is: [Yes/No]**")
        
        return "\n".join(prompt_parts)
    
    def generate(
        self,
        question: str,
        context: str,
        choices: Optional[List[str]] = None,
        domain: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate an answer using SEF prompting.
        
        Args:
            question: The question to answer
            context: The context/passage
            choices: Optional list of answer choices
            domain: Optional domain override
            
        Returns:
            Dictionary with answer and explanation
        """
        # Allow domain override for this call
        if domain:
            original_domain = self.domain
            self.domain = domain
        
        # Build prompt
        prompt = self._build_prompt(context, question, choices)
        
        # Get response from LLM
        response = self.llm_client.generate(prompt)
        
        # Extract answer
        predicted_answer = extract_answer(
            response=response,
            choices=choices,
        )
        
        # Restore domain if overridden
        if domain:
            self.domain = original_domain
        
        return {
            'answer': predicted_answer,
            'explanation': response,
            'method': self.name,
            'domain': self.domain,
            'components': {
                'answer_preview': self.use_answer_preview,
                'answer_clarity': self.use_answer_clarity,
                'conclusion_isolation': self.use_conclusion_isolation,
                'domain_terminology': self.use_domain_terminology,
                'evidence_alignment': self.use_evidence_alignment,
                'fact_specificity': self.use_fact_specificity,
            }
        }


class SEFAblation(SEFPrompting):
    """
    SEF ablation variants for studying component contributions.
    
    Ablation conditions:
    - full: All 6 components
    - no_afl: Without Answer First/Last
    - no_ac: Without Answer Clarity
    - no_ci: Without Conclusion Isolation
    - no_dtc: Without Domain Terminology Consistency
    - no_cea: Without Conclusion Evidence Alignment
    - no_fs: Without Fact Specificity
    """
    
    ABLATION_CONFIGS = {
        'full': {},
        'no_afl': {'answer_preview': False},  # Remove answer preview (AFL)
        'no_ac': {'answer_clarity': False},  # Remove answer clarity (AC)
        'no_ci': {'conclusion_isolation': False},  # Remove conclusion isolation (CI)
        'no_dtc': {'domain_terminology': False},  # Remove domain terminology (DTC)
        'no_cea': {'evidence_alignment': False},  # Remove evidence alignment (CEA)
        'no_fs': {'fact_specificity': False},  # Remove fact specificity (FS)
        # Combined ablations
        'no_presentation': {  # Remove all presentation components
            'answer_preview': False,
            'answer_clarity': False,
            'conclusion_isolation': False,
        },
        'no_domain': {  # Remove all domain reasoning components
            'domain_terminology': False,
            'evidence_alignment': False,
            'fact_specificity': False,
        },
    }
    
    def __init__(
        self, 
        llm_client, 
        ablation_type: str = 'full',
        domain: str = 'general',
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize SEF ablation variant.
        
        Args:
            llm_client: LLM client
            ablation_type: One of ABLATION_CONFIGS keys
            domain: Domain for terminology
            config: Additional configuration
        """
        if ablation_type not in self.ABLATION_CONFIGS:
            raise ValueError(f"Unknown ablation type: {ablation_type}. "
                           f"Available: {list(self.ABLATION_CONFIGS.keys())}")
        
        # Merge ablation config with user config
        ablation_config = self.ABLATION_CONFIGS[ablation_type].copy()
        if config:
            ablation_config.update(config)
        
        super().__init__(llm_client, domain, ablation_config)
        self.name = f"sef_{ablation_type}"
        self.ablation_type = ablation_type
        
        logger.info(f"Initialized SEFAblation: {ablation_type} for domain={domain}")

