"""
Vanilla RAG Baseline
Basic retrieve-then-generate approach
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class VanillaRAG:
    """
    Vanilla RAG baseline - retrieve relevant statute, then generate.
    """
    
    def __init__(self, llm_client, top_k: int = 3):
        """
        Initialize Vanilla RAG baseline.
        
        Args:
            llm_client: LLM client (OpenAI, Anthropic, or Google)
            top_k: Number of passages to retrieve
        """
        self.llm_client = llm_client
        self.top_k = top_k
        self.name = "vanilla_rag"
        logger.info(f"Initialized VanillaRAG baseline with top_k={top_k}")
    
    def generate(
        self,
        question: str,
        context: str,
        choices: Optional[list] = None,
        statute_chunks: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate an answer using vanilla RAG.
        
        Args:
            question: The question
            context: The context
            choices: Optional list of answer choices
            statute_chunks: Pre-chunked statute text (if available)
            
        Returns:
            Dictionary with answer and explanation
        """
        # Step 1: Retrieve relevant passages
        if statute_chunks:
            retrieved = self._retrieve(question, statute_chunks)
        else:
            # Chunk the context
            chunks = self._chunk_text(context)
            retrieved = self._retrieve(question, chunks)
        
        # Step 2: Generate with retrieved context
        retrieved_context = "\n\n".join(retrieved)
        
        if choices:
            choices_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
            prompt = f"""Answer the following question using the provided context.

Retrieved Context:
{retrieved_context}

Question: {question}

Choices:
{choices_text}

Based on the retrieved context, provide your reasoning and answer.

Reasoning:"""
        else:
            prompt = f"""Answer the following question using the provided context.

Retrieved Context:
{retrieved_context}

Question: {question}

Based on the retrieved context, provide your reasoning and answer (Yes or No).

Reasoning:"""
        
        response = self.llm_client.generate(prompt)
        answer = self._extract_answer(response, choices)
        
        return {
            "method": self.name,
            "answer": answer,
            "explanation": response,
            "raw_response": response,
            "retrieved_passages": retrieved,
        }
    
    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into chunks."""
        # Simple sentence-based chunking
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
        """
        Retrieve relevant chunks (simple keyword-based for baseline).
        In production, use vector similarity search.
        """
        # Simple keyword overlap scoring
        question_words = set(question.lower().split())
        
        scored_chunks = []
        for chunk in chunks:
            chunk_words = set(chunk.lower().split())
            overlap = len(question_words & chunk_words)
            scored_chunks.append((overlap, chunk))
        
        # Sort by score and return top k
        scored_chunks.sort(reverse=True)
        return [chunk for _, chunk in scored_chunks[:self.top_k]]
    
    def _extract_answer(self, response: str, choices: Optional[list]) -> str:
        """Extract the answer from the response."""
        from src.utils.answer_extractor import extract_answer
        return extract_answer(response, choices=choices)

