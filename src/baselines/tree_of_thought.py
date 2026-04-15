"""
Tree-of-Thought (ToT) Baseline
Branching reasoning paths with evaluation
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class TreeOfThought:
    """
    Tree-of-Thought reasoning baseline.
    Explores multiple reasoning branches and evaluates them.
    """
    
    def __init__(self, llm_client, num_branches: int = 3):
        """
        Initialize ToT baseline.
        
        Args:
            llm_client: LLM client (OpenAI, Anthropic, or Google)
            num_branches: Number of reasoning branches to explore
        """
        self.llm_client = llm_client
        self.num_branches = num_branches
        self.name = "tree_of_thought"
        logger.info(f"Initialized TreeOfThought baseline with {num_branches} branches")
    
    def generate(
        self,
        question: str,
        context: str,
        choices: Optional[list] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate an answer using tree-of-thought reasoning.
        
        Args:
            question: The question
            context: The context
            choices: Optional list of answer choices
            
        Returns:
            Dictionary with answer and explanation
        """
        # Step 1: Generate multiple initial thoughts
        initial_thoughts = self._generate_initial_thoughts(question, context, choices)
        
        # Step 2: Develop each thought into a full reasoning path
        reasoning_paths = []
        for thought in initial_thoughts:
            path = self._develop_thought(thought, question, context, choices)
            reasoning_paths.append(path)
        
        # Step 3: Evaluate and select best path
        best_path = self._evaluate_paths(reasoning_paths, question, context)
        
        return {
            "method": self.name,
            "answer": best_path["answer"],
            "explanation": best_path["reasoning"],
            "raw_response": best_path["reasoning"],
            "all_paths": reasoning_paths,
            "selected_path": best_path,
        }
    
    def _generate_initial_thoughts(
        self,
        question: str,
        context: str,
        choices: Optional[list]
    ) -> List[str]:
        """Generate diverse initial reasoning approaches."""
        prompt = f"""Given this question, propose {self.num_branches} different approaches to analyze it.

Context:
{context}

Question: {question}

List {self.num_branches} different analytical approaches (each 1-2 sentences):
1."""
        
        response = self.llm_client.generate(prompt, temperature=0.8)
        
        # Parse thoughts
        thoughts = []
        lines = response.split('\n')
        current_thought = ""
        
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                if current_thought:
                    thoughts.append(current_thought)
                current_thought = line.lstrip('0123456789.-) ')
            elif current_thought:
                current_thought += " " + line
        
        if current_thought:
            thoughts.append(current_thought)
        
        # Ensure we have enough thoughts
        while len(thoughts) < self.num_branches:
            thoughts.append(f"Approach {len(thoughts)+1}: Analyze the key elements systematically.")
        
        return thoughts[:self.num_branches]
    
    def _develop_thought(
        self,
        initial_thought: str,
        question: str,
        context: str,
        choices: Optional[list]
    ) -> Dict[str, Any]:
        """Develop an initial thought into a full reasoning path."""
        if choices:
            choices_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
            prompt = f"""Develop the following analytical approach into a complete analysis.

Context:
{context}

Question: {question}

Choices:
{choices_text}

Analytical Approach: {initial_thought}

Provide a complete analysis following this approach, then state your final answer.

Analysis:"""
        else:
            prompt = f"""Develop the following analytical approach into a complete analysis.

Context:
{context}

Question: {question}

Analytical Approach: {initial_thought}

Provide a complete analysis following this approach, then state your final answer (Yes or No).

Analysis:"""
        
        response = self.llm_client.generate(prompt)
        answer = self._extract_answer(response, choices)
        
        return {
            "initial_thought": initial_thought,
            "reasoning": response,
            "answer": answer,
        }
    
    def _evaluate_paths(
        self,
        paths: List[Dict[str, Any]],
        question: str,
        context: str
    ) -> Dict[str, Any]:
        """Evaluate reasoning paths and select the best one."""
        if not paths:
            return {"answer": "", "reasoning": "", "score": 0}
        
        # Build evaluation prompt
        paths_text = ""
        for i, path in enumerate(paths):
            paths_text += f"\nPath {i+1}:\n{path['reasoning'][:500]}...\nAnswer: {path['answer']}\n"
        
        eval_prompt = f"""Evaluate the following reasoning paths for this question and select the best one.

Question: {question}

{paths_text}

Which path (1, 2, or 3) provides the most thorough and accurate analysis? 
Respond with just the number."""
        
        response = self.llm_client.generate(eval_prompt)
        
        # Parse selection
        try:
            for char in response:
                if char.isdigit():
                    idx = int(char) - 1
                    if 0 <= idx < len(paths):
                        return paths[idx]
        except:
            pass
        
        # Default to first path
        return paths[0]
    
    def _extract_answer(self, response: str, choices: Optional[list]) -> str:
        """Extract the answer from a reasoning path."""
        from src.utils.answer_extractor import extract_answer
        return extract_answer(response, choices=choices)
        
        return ""

