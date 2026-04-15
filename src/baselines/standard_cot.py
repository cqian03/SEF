"""
Standard Chain-of-Thought (CoT) Baseline
"Let's think step by step"
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class StandardCoT:
    """
    Standard Chain-of-Thought prompting baseline.
    Uses "Let's think step by step" trigger.
    """
    
    def __init__(self, llm_client):
        """
        Initialize CoT baseline.
        
        Args:
            llm_client: LLM client (OpenAI, Anthropic, or Google)
        """
        self.llm_client = llm_client
        self.name = "standard_cot"
        logger.info("Initialized StandardCoT baseline")
    
    def generate(
        self,
        question: str,
        context: str,
        choices: Optional[list] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate an answer using chain-of-thought prompting.
        
        Args:
            question: The question
            context: The context
            choices: Optional list of answer choices
            
        Returns:
            Dictionary with answer and explanation
        """
        # Build prompt with CoT trigger
        if choices:
            choices_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
            prompt = f"""Given the following context and question, let's think step by step.

Context:
{context}

Question: {question}

Choices:
{choices_text}

Let's think step by step:
1. First, let me understand the context and relevant rules.
2. Then, I'll analyze how they apply to this question.
3. Finally, I'll determine the correct answer.

Step-by-step reasoning:"""
        else:
            prompt = f"""Given the following context and question, let's think step by step.

Context:
{context}

Question: {question}

Let's think step by step:
1. First, let me understand the context and relevant rules.
2. Then, I'll analyze how they apply to this question.
3. Finally, I'll determine the correct answer.

Step-by-step reasoning:"""
        
        # Generate response
        response = self.llm_client.generate(prompt)
        
        # Parse answer and explanation
        answer, explanation = self._parse_response(response, choices)
        
        return {
            "method": self.name,
            "answer": answer,
            "explanation": explanation,
            "raw_response": response,
        }
    
    def _parse_response(self, response: str, choices: Optional[list] = None) -> tuple:
        """Parse the answer and explanation from the response."""
        explanation = response
        answer = ""
        
        # Look for explicit answer markers
        response_lower = response.lower()
        
        # Try to find answer at the end
        lines = response.split('\n')
        for line in reversed(lines):
            line_lower = line.lower().strip()
            
            if "answer:" in line_lower or "therefore" in line_lower or "conclusion" in line_lower:
                if "yes" in line_lower:
                    answer = "Yes"
                    break
                elif "no" in line_lower:
                    answer = "No"
                    break
                elif choices:
                    for i in range(len(choices)):
                        if str(i+1) in line_lower or str(i) in line_lower:
                            answer = str(i)
                            break
        
        # Fallback: check entire response
        if not answer:
            if "yes" in response_lower[-100:]:
                answer = "Yes"
            elif "no" in response_lower[-100:]:
                answer = "No"
        
        return answer, explanation

