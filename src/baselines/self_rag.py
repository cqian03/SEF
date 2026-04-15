"""
Self-RAG Baseline
Retrieval with self-reflection and adaptive retrieval
Reference: Asai et al., ICLR 2024
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class SelfRAG:
    """
    Self-RAG baseline - retrieval with self-reflection.
    Decides when to retrieve and reflects on retrieved content.
    """
    
    def __init__(self, llm_client, top_k: int = 3):
        """
        Initialize Self-RAG baseline.
        
        Args:
            llm_client: LLM client (OpenAI, Anthropic, or Google)
            top_k: Number of passages to retrieve
        """
        self.llm_client = llm_client
        self.top_k = top_k
        self.name = "self_rag"
        logger.info(f"Initialized SelfRAG baseline with top_k={top_k}")
    
    def generate(
        self,
        question: str,
        context: str,
        choices: Optional[list] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate an answer using Self-RAG.
        
        Args:
            question: The question
            context: The context
            choices: Optional list of answer choices
            
        Returns:
            Dictionary with answer and explanation
        """
        # Step 1: Decide if retrieval is needed
        needs_retrieval = self._assess_retrieval_need(question, context)
        
        if needs_retrieval:
            # Step 2: Retrieve relevant passages
            chunks = self._chunk_text(context)
            retrieved = self._retrieve(question, chunks)
            
            # Step 3: Assess relevance of retrieved passages
            relevant_passages = self._assess_relevance(question, retrieved)
            
            # Step 4: Generate with relevant context
            generation_context = "\n\n".join(relevant_passages)
        else:
            generation_context = context
            retrieved = []
            relevant_passages = []
        
        # Step 5: Generate response
        response, answer = self._generate_with_reflection(
            question, generation_context, choices
        )
        
        # Step 6: Self-critique the response
        critique = self._critique_response(question, response, context)
        
        # Step 7: Refine if needed
        if critique["needs_refinement"]:
            response, answer = self._refine_response(
                question, response, critique, context, choices
            )
        
        return {
            "method": self.name,
            "answer": answer,
            "explanation": response,
            "raw_response": response,
            "retrieval_decision": needs_retrieval,
            "retrieved_passages": retrieved,
            "relevant_passages": relevant_passages,
            "critique": critique,
        }
    
    def _assess_retrieval_need(self, question: str, context: str) -> bool:
        """Assess if retrieval is needed for this question."""
        prompt = f"""Determine if additional context retrieval is needed to answer this question.

Question: {question}

Available Context Length: {len(context)} characters

Do we need to retrieve specific passages, or can we use the full context?
Answer with just: RETRIEVE or USE_FULL"""
        
        response = self.llm_client.generate(prompt)
        return "RETRIEVE" in response.upper()
    
    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into chunks."""
        sentences = text.replace('\n', ' ').split('.')
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence + "."
            else:
                current_chunk += " " + sentence + "."
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks if chunks else [text]
    
    def _retrieve(self, question: str, chunks: List[str]) -> List[str]:
        """Retrieve relevant chunks."""
        question_words = set(question.lower().split())
        
        scored_chunks = []
        for chunk in chunks:
            chunk_words = set(chunk.lower().split())
            overlap = len(question_words & chunk_words)
            scored_chunks.append((overlap, chunk))
        
        scored_chunks.sort(reverse=True)
        return [chunk for _, chunk in scored_chunks[:self.top_k]]
    
    def _assess_relevance(self, question: str, passages: List[str]) -> List[str]:
        """Assess relevance of each retrieved passage."""
        relevant = []
        
        for passage in passages:
            prompt = f"""Is this passage relevant for answering the question?

Question: {question}

Passage: {passage[:300]}...

Answer with just: RELEVANT or NOT_RELEVANT"""
            
            response = self.llm_client.generate(prompt)
            if "RELEVANT" in response.upper() and "NOT" not in response.upper():
                relevant.append(passage)
        
        # Return at least one passage
        return relevant if relevant else passages[:1]
    
    def _generate_with_reflection(
        self,
        question: str,
        context: str,
        choices: Optional[list]
    ) -> tuple:
        """Generate response with self-reflection tokens."""
        if choices:
            choices_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
            prompt = f"""Answer this question using the provided context.

Context:
{context}

Question: {question}

Choices:
{choices_text}

[Reflection: Assess if the context is sufficient]
[Generation: Provide reasoning]
[Answer: State your final answer]

Response:"""
        else:
            prompt = f"""Answer this question using the provided context.

Context:
{context}

Question: {question}

[Reflection: Assess if the context is sufficient]
[Generation: Provide reasoning]
[Answer: State your final answer (Yes or No)]

Response:"""
        
        response = self.llm_client.generate(prompt)
        answer = self._extract_answer(response, choices)
        
        return response, answer
    
    def _critique_response(
        self,
        question: str,
        response: str,
        context: str
    ) -> Dict[str, Any]:
        """Self-critique the generated response."""
        prompt = f"""Critique this analysis for accuracy and completeness.

Question: {question}

Analysis:
{response[:500]}

Critique the analysis:
1. Is it factually supported by the context?
2. Does it address the key elements?
3. Is the reasoning sound?

Provide brief critique and indicate if refinement is needed.
End with: NEEDS_REFINEMENT or SUFFICIENT"""
        
        critique_response = self.llm_client.generate(prompt)
        
        return {
            "critique": critique_response,
            "needs_refinement": "NEEDS_REFINEMENT" in critique_response.upper()
        }
    
    def _refine_response(
        self,
        question: str,
        original_response: str,
        critique: Dict[str, Any],
        context: str,
        choices: Optional[list]
    ) -> tuple:
        """Refine the response based on critique."""
        if choices:
            choices_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
            prompt = f"""Improve this analysis based on the critique.

Context:
{context}

Question: {question}

Choices:
{choices_text}

Original Analysis:
{original_response[:300]}

Critique:
{critique['critique'][:200]}

Provide an improved analysis addressing the critique issues.

Improved Analysis:"""
        else:
            prompt = f"""Improve this analysis based on the critique.

Context:
{context}

Question: {question}

Original Analysis:
{original_response[:300]}

Critique:
{critique['critique'][:200]}

Provide an improved analysis addressing the critique issues.

Improved Analysis:"""
        
        refined_response = self.llm_client.generate(prompt)
        answer = self._extract_answer(refined_response, choices)
        
        return refined_response, answer
    
    def _extract_answer(self, response: str, choices: Optional[list]) -> str:
        """Extract the answer from the response."""
        from src.utils.answer_extractor import extract_answer
        return extract_answer(response, choices=choices)

