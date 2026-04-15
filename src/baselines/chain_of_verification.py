"""
Chain-of-Verification (CoV) Baseline
Self-verification step after initial generation
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ChainOfVerification:
    """
    Chain-of-Verification baseline.
    Generates initial answer, then verifies and potentially revises.
    """
    
    def __init__(self, llm_client):
        """
        Initialize CoV baseline.
        
        Args:
            llm_client: LLM client (OpenAI, Anthropic, or Google)
        """
        self.llm_client = llm_client
        self.name = "chain_of_verification"
        logger.info("Initialized ChainOfVerification baseline")
    
    def generate(
        self,
        question: str,
        context: str,
        choices: Optional[list] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate an answer with self-verification.
        
        Args:
            question: The question
            context: The context
            choices: Optional list of answer choices
            
        Returns:
            Dictionary with answer and explanation
        """
        # Step 1: Generate initial response
        initial_response = self._generate_initial(question, context, choices)
        
        # Step 2: Generate verification questions
        verification_questions = self._generate_verification_questions(
            question, context, initial_response
        )
        
        # Step 3: Answer verification questions
        verification_answers = self._answer_verification_questions(
            verification_questions, context
        )
        
        # Step 4: Final verified response
        final_response = self._generate_final(
            question, context, choices, initial_response, verification_answers
        )
        
        return {
            "method": self.name,
            "answer": final_response["answer"],
            "explanation": final_response["explanation"],
            "raw_response": final_response["explanation"],
            "initial_response": initial_response,
            "verification_questions": verification_questions,
            "verification_answers": verification_answers,
        }
    
    def _generate_initial(
        self,
        question: str,
        context: str,
        choices: Optional[list]
    ) -> Dict[str, str]:
        """Generate initial response."""
        if choices:
            choices_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
            prompt = f"""Answer the following question with reasoning.

Context:
{context}

Question: {question}

Choices:
{choices_text}

Reasoning and Answer:"""
        else:
            prompt = f"""Answer the following question with reasoning.

Context:
{context}

Question: {question}

Reasoning and Answer:"""
        
        response = self.llm_client.generate(prompt)
        answer = self._extract_answer(response, choices)
        
        return {"reasoning": response, "answer": answer}
    
    def _generate_verification_questions(
        self,
        question: str,
        context: str,
        initial_response: Dict[str, str]
    ) -> list:
        """Generate questions to verify the initial response."""
        prompt = f"""Given this analysis, generate 3 verification questions to check its accuracy.

Original Question: {question}

Initial Analysis:
{initial_response['reasoning'][:500]}

Generate 3 specific questions to verify the key claims in this analysis:
1."""
        
        response = self.llm_client.generate(prompt)
        
        # Parse questions
        questions = []
        for line in response.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                q = line.lstrip('0123456789.-) ')
                if q:
                    questions.append(q)
        
        return questions[:3] if questions else [
            "Is the relevant rule correctly identified?",
            "Are all required elements properly analyzed?",
            "Does the conclusion follow from the analysis?"
        ]
    
    def _answer_verification_questions(
        self,
        questions: list,
        context: str
    ) -> list:
        """Answer each verification question."""
        answers = []
        
        for question in questions:
            prompt = f"""Answer this verification question based on the context.

Context:
{context}

Verification Question: {question}

Brief Answer:"""
            
            response = self.llm_client.generate(prompt)
            answers.append({"question": question, "answer": response})
        
        return answers
    
    def _generate_final(
        self,
        question: str,
        context: str,
        choices: Optional[list],
        initial_response: Dict[str, str],
        verification_answers: list
    ) -> Dict[str, str]:
        """Generate final verified response."""
        verification_text = "\n".join([
            f"Q: {va['question']}\nA: {va['answer']}"
            for va in verification_answers
        ])
        
        if choices:
            choices_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
            prompt = f"""Based on your initial analysis and verification, provide your final answer.

Context:
{context}

Question: {question}

Choices:
{choices_text}

Initial Analysis:
{initial_response['reasoning'][:300]}

Verification Results:
{verification_text}

Final Analysis and Answer (incorporate any corrections from verification):"""
        else:
            prompt = f"""Based on your initial analysis and verification, provide your final answer.

Context:
{context}

Question: {question}

Initial Analysis:
{initial_response['reasoning'][:300]}

Verification Results:
{verification_text}

Final Analysis and Answer (incorporate any corrections from verification):"""
        
        response = self.llm_client.generate(prompt)
        answer = self._extract_answer(response, choices)
        
        return {"explanation": response, "answer": answer}
    
    def _extract_answer(self, response: str, choices: Optional[list]) -> str:
        """Extract the answer from a response."""
        from src.utils.answer_extractor import extract_answer
        return extract_answer(response, choices=choices)

