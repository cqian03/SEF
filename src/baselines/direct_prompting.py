"""
Direct Prompting Baseline: Zero-shot, no explanation requested.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class DirectPrompting:
    """
    Direct prompting baseline - asks for answer without explicit reasoning.
    """
    
    def __init__(self, llm_client):
        """
        Initialize direct prompting baseline.
        
        Args:
            llm_client: LLM client (OpenAI, Anthropic, or Google)
        """
        self.llm_client = llm_client
        self.name = "direct"
        logger.info("Initialized DirectPrompting baseline")
    
    def generate(
        self,
        question: str,
        context: str,
        choices: Optional[list] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate an answer using direct prompting.
        
        Args:
            question: The question
            context: The context
            choices: Optional list of answer choices
            
        Returns:
            Dictionary with answer and explanation
        """
        # Build prompt
        if choices:
            choices_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
            prompt = f"""Answer the following question.

Context:
{context}

Question: {question}

Choices:
{choices_text}

Provide only the answer (the choice number or Yes/No).
Answer:"""
        else:
            prompt = f"""Answer the following question.

Context:
{context}

Question: {question}

Provide only the answer (Yes/No).
Answer:"""
        
        # Generate response
        response = self.llm_client.generate(prompt)
        
        # Parse answer
        answer = self._parse_answer(response, choices)
        
        return {
            "method": self.name,
            "answer": answer,
            "explanation": "",  # Direct prompting has no explanation
            "raw_response": response,
        }
    
    def _parse_answer(self, response: str, choices: Optional[list] = None) -> str:
        """Parse the answer from the response.
        
        For binary Yes/No tasks, always returns "Yes" or "No".
        For multi-choice tasks, returns 1-indexed choice number.
        """
        response = response.strip().lower()
        
        # Check for explicit Yes/No text
        if "yes" in response:
            return "Yes"
        elif "no" in response:
            return "No"
        
        # Check for choice numbers
        if choices:
            # For binary Yes/No choices, map numbers back to Yes/No
            if len(choices) == 2 and set(c.lower() for c in choices) == {'yes', 'no'}:
                if "1" in response:
                    # Find which choice is at index 0
                    return choices[0]  # "Yes" or "No"
                elif "2" in response:
                    return choices[1]  # "Yes" or "No"
            else:
                # Multi-choice: return 1-indexed number
                for i in range(1, len(choices) + 1):
                    if str(i) in response:
                        return str(i)
        
        return response

