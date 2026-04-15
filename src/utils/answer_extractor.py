"""
Answer Extractor for SEF

Extracts answers based on task type:
- Binary (Yes/No) tasks: hearsay, consumer_contracts_qa, pubmedqa, fpb

Note: All methods return 1-indexed for multi-choice. Evaluator converts to 0-indexed for comparison.
"""

import re
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


def extract_answer(
    response: str,
    choices: Optional[List[str]] = None,
    task_name: Optional[str] = None
) -> str:
    """
    Extract the answer from LLM response.
    
    For multi-choice tasks (casehold), prioritize finding choice numbers.
    For binary tasks, look for Yes/No.
    
    Args:
        response: The LLM response text
        choices: List of answer choices (if multi-choice task)
        task_name: Name of the task (optional, for task-specific logic)
        
    Returns:
        Extracted answer string
    """
    if not response:
        return ""
    
    response_text = response.strip()
    
    # Multi-choice task (casehold has 5 choices)
    if choices and len(choices) > 2:
        return _extract_multichoice_answer(response_text, len(choices), choices)
    
    # Binary task (Yes/No)
    return _extract_binary_answer(response_text)


def _extract_multichoice_answer(
    response: str, 
    num_choices: int, 
    choices: Optional[List[str]] = None
) -> str:
    """
    Extract answer for multi-choice tasks.
    
    Returns 1-indexed answer (1 to num_choices) matching how choices are shown to model.
    For CaseHold with 5 choices, returns "1" to "5".
    
    Enhanced patterns for various response formats including:
    - "The answer is 4"
    - "Therefore, 2."
    - "Option 3" / "Choice 1"
    - "Holding 2" (legal context)
    - Ordinal: "the first/second/third..."
    - Letter choices: (a), (b), (c)...
    - Boxed answers: [3], **3**
    - Choice content matching (when model outputs the choice text)
    """
    # Use last 800 chars for better context
    end_text = response[-800:].lower()
    full_text = response.lower()
    
    # Helper to validate and return 1-indexed answer
    def to_1indexed(num: int) -> str:
        """Convert any digit to 1-indexed if valid."""
        if 1 <= num <= num_choices:
            return str(num)
        elif 0 <= num < num_choices:
            return str(num + 1)  # Convert 0-indexed to 1-indexed
        return ""
    
    # ===== STRONG PATTERNS (High confidence) =====
    
    # Pattern 1: Explicit answer statements
    strong_patterns = [
        r'the (?:correct )?answer is[:\s]*[(\[]?(\d)[)\]]?',
        r'(?:my |final )?answer[:\s]+[(\[]?(\d)[)\]]?',
        r'correct (?:answer|choice|option) is[:\s]*[(\[]?(\d)[)\]]?',
        r'i (?:would )?(?:choose|select|pick)[:\s]*[(\[]?(\d)[)\]]?',
        r'best (?:answer|choice|option) is[:\s]*[(\[]?(\d)[)\]]?',
    ]
    
    for pattern in strong_patterns:
        match = re.search(pattern, end_text)
        if match:
            result = to_1indexed(int(match.group(1)))
            if result:
                return result
    
    # Pattern 2: Legal-specific patterns (CaseHold uses "holding")
    legal_patterns = [
        r'holding[:\s]*[(\[]?(\d)[)\]]?',
        r'holding (?:number |#)?(\d)',
        r'(?:the )?(?:correct |best |most appropriate )?holding is[:\s]*[(\[]?(\d)[)\]]?',
    ]
    
    for pattern in legal_patterns:
        match = re.search(pattern, end_text)
        if match:
            result = to_1indexed(int(match.group(1)))
            if result:
                return result
    
    # Pattern 3: Conclusion markers with numbers
    conclusion_patterns = [
        r'(?:therefore|thus|hence|so|accordingly)[,:\s]+[(\[]?(\d)[)\]]?',
        r'(?:in )?conclusion[,:\s]+[(\[]?(\d)[)\]]?',
        r'final(?:ly)?[,:\s]+[(\[]?(\d)[)\]]?',
    ]
    
    for pattern in conclusion_patterns:
        match = re.search(pattern, end_text)
        if match:
            result = to_1indexed(int(match.group(1)))
            if result:
                return result
    
    # Pattern 4: Option/Choice explicit mentions
    option_patterns = [
        r'(?:option|choice|alternative)[:\s]*[(\[]?(\d)[)\]]?',
        r'(?:option|choice|alternative) (?:number |#)?(\d)',
        r'\((\d)\) is (?:the )?(?:correct|best|right)',
        r'(?:select|pick|choose) (?:option |choice )?[(\[]?(\d)[)\]]?',
    ]
    
    for pattern in option_patterns:
        match = re.search(pattern, end_text)
        if match:
            result = to_1indexed(int(match.group(1)))
            if result:
                return result
    
    # ===== MEDIUM PATTERNS =====
    
    # Pattern 5: Ordinal numbers (first, second, etc.)
    ordinal_map = {
        'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
        '1st': 1, '2nd': 2, '3rd': 3, '4th': 4, '5th': 5,
    }
    ordinal_pattern = r'the (' + '|'.join(ordinal_map.keys()) + r') (?:option|choice|holding|one)'
    match = re.search(ordinal_pattern, end_text)
    if match:
        ordinal = match.group(1)
        num = ordinal_map.get(ordinal)
        if num and 1 <= num <= num_choices:
            return str(num)
    
    # Pattern 6: Letter choices (a, b, c, d, e) -> convert to numbers
    letter_map = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
    letter_patterns = [
        r'the answer is[:\s]*\(?([a-e])\)?',
        r'answer[:\s]+\(?([a-e])\)?',
        r'(?:option|choice)[:\s]*\(?([a-e])\)?',
    ]
    for pattern in letter_patterns:
        match = re.search(pattern, end_text)
        if match:
            letter = match.group(1)
            num = letter_map.get(letter)
            if num and num <= num_choices:
                return str(num)
    
    # Pattern 7: Boxed or emphasized answers
    boxed_patterns = [
        r'\*\*(\d)\*\*',  # **3**
        r'\[(\d)\]',      # [3]
        r'`(\d)`',        # `3`
        r'「(\d)」',       # Japanese brackets
    ]
    for pattern in boxed_patterns:
        matches = re.findall(pattern, end_text)
        if matches:
            # Take the last boxed number
            for num_str in reversed(matches):
                result = to_1indexed(int(num_str))
                if result:
                    return result
    
    # ===== WEAK PATTERNS (Lower confidence, use last 200 chars) =====
    
    last_200 = response[-200:].lower()
    
    # Pattern 8: Digit at start of final lines
    lines = response.split('\n')
    for line in reversed(lines[-5:]):
        line = line.strip()
        # Match "4." or "4:" or "4)" or just "4" at start
        match = re.match(r'^[(\[]?(\d)[)\].\s:]', line)
        if match:
            result = to_1indexed(int(match.group(1)))
            if result:
                return result
    
    # Pattern 9: Standalone digit near the end (very end only)
    last_50 = response[-50:]
    # Look for pattern like "... is 3." or "... 3." at very end
    match = re.search(r'(\d)[.\s]*$', last_50)
    if match:
        result = to_1indexed(int(match.group(1)))
        if result:
            return result
    
    # Pattern 10: Any valid digit in last 100 chars (last resort)
    last_100 = response[-100:]
    digits = re.findall(r'\b(\d)\b', last_100)
    if digits:
        # Prefer digits that appear after conclusion words
        for d in reversed(digits):
            num = int(d)
            result = to_1indexed(num)
            if result:
                return result
    
    # Pattern 11: Search full text for very explicit patterns
    full_strong_patterns = [
        r'(?:my )?(?:final )?answer[:\s]+[(\[]?(\d)[)\]]?[.\s]*$',
        r'correct answer[:\s]+[(\[]?(\d)[)\]]?',
    ]
    for pattern in full_strong_patterns:
        match = re.search(pattern, full_text)
        if match:
            result = to_1indexed(int(match.group(1)))
            if result:
                return result
    
    # ===== CHOICE CONTENT MATCHING (when model outputs choice text) =====
    # This handles cases where the model says "the answer is: holding that..."
    # instead of giving a number
    
    if choices:
        # Check if response contains or closely matches any choice text
        response_lower = response.lower()
        
        # Method 1: Check if response ends with (or contains near end) a choice's key phrase
        # Extract significant phrases from each choice (>15 chars, not common legal words)
        best_match_idx = -1
        best_match_score = 0
        
        for idx, choice in enumerate(choices):
            choice_lower = choice.lower().strip()
            
            # Skip very short choices
            if len(choice_lower) < 20:
                continue
            
            # Check if choice text appears in last 500 chars of response
            last_500 = response_lower[-500:]
            
            # Full choice match
            if choice_lower in last_500:
                # Calculate match score based on position (later = better)
                pos = last_500.rfind(choice_lower)
                score = pos + len(choice_lower) * 2  # Prefer longer matches at end
                if score > best_match_score:
                    best_match_score = score
                    best_match_idx = idx
                continue
            
            # Partial match: check if significant portion of choice is in response
            # Split choice into words and check overlap
            choice_words = set(choice_lower.split())
            # Remove common legal words that appear in many choices
            common_words = {'the', 'a', 'an', 'that', 'which', 'is', 'are', 'was', 'were', 
                          'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                          'holding', 'held', 'court', 'defendant', 'plaintiff', 'case',
                          'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'or', 'and'}
            significant_words = choice_words - common_words
            
            if len(significant_words) >= 3:
                # Count how many significant words appear in last 300 chars
                last_300 = response_lower[-300:]
                matches = sum(1 for w in significant_words if w in last_300)
                match_ratio = matches / len(significant_words)
                
                if match_ratio >= 0.6:  # At least 60% of significant words match
                    score = match_ratio * 100 + len(significant_words)
                    if score > best_match_score:
                        best_match_score = score
                        best_match_idx = idx
        
        if best_match_idx >= 0:
            # Return 1-indexed
            return str(best_match_idx + 1)
    
    logger.warning(f"Could not extract multi-choice answer from response (last 100 chars: {response[-100:]!r})")
    return ""


def _extract_binary_answer(response: str) -> str:
    """
    Extract Yes/No answer for binary tasks.
    
    Looks for explicit Yes/No patterns, avoiding false matches.
    Handles various formats including:
    - "Answer: Yes"
    - "**Answer:**\n1. Yes"
    - "Final Answer:\n2. No"
    - "Therefore, Yes"
    - "The correct answer is:\n1. Yes"
    - "**1. Yes**"
    - "Choice 2: No"
    """
    # Check last 600 chars for better context
    response_lower = response.lower()
    
    # Use unified patterns that capture yes/no - order matters
    patterns = [
        # "My answer is: Yes" or "My answer is: **Yes**" or "My answer is: **No**."
        r'[Mm]y answer is[:\s]*\*?\*?(?:\d+[.\)]\s*)?([Yy]es|[Nn]o)\b',
        # "**Final Answer: 2. No**" or "Final Answer:\n1. Yes"
        r'[Ff]inal [Aa]nswer[:\s]*\*?\*?(?:\d+[.\)]\s*)?([Yy]es|[Nn]o)\b',
        r'\*\*[Ff]inal [Aa]nswer[:\s]*(?:\d+[.\)]\s*)?([Yy]es|[Nn]o)\b',
        # "The correct answer is:\n1. Yes" or "correct answer is: No"
        r'correct answer (?:is|would be)[:\s]*\*?\*?(?:\d+[.\)]\s*)?([Yy]es|[Nn]o)\b',
        # "The answer is:\n1. Yes" or "answer is: No"
        r'the answer (?:is|would be)[:\s]*\*?\*?(?:\d+[.\)]\s*)?([Yy]es|[Nn]o)\b',
        # "**Answer:**\n1. Yes" or "Answer: No"
        r'\*\*[Aa]nswer\*?\*?[:\s]*\n?\s*(?:\d+[.\)]\s*)?([Yy]es|[Nn]o)\b',
        r'[Aa]nswer[:\s]+\*?\*?(?:\d+[.\)]\s*)?([Yy]es|[Nn]o)\b',
        # "**1. Yes**" or "**2. No**" (bold numbered)
        r'\*\*\d+[.\)]\s*([Yy]es|[Nn]o)\*\*',
        # "1. Yes" or "2. No" standalone on line (not bold)
        r'(?:^|\n)\s*\d+[.\)]\s*([Yy]es|[Nn]o)\s*(?:\n|$)',
        # "Choice 1: Yes" or "Choice 2 (No)"
        r'[Cc]hoice\s*\d+[:\s]*\(?([Yy]es|[Nn]o)\)?',
        # "**Yes**" or "**No**" standalone
        r'\*\*([Yy]es|[Nn]o)\*\*',
        # "Therefore, yes" or "Thus, no"
        r'[Tt]herefore[,:\s]+(?:the answer (?:is|would be)[:\s]*)?([Yy]es|[Nn]o)\b',
        r'[Tt]hus[,:\s]+(?:the answer (?:is|would be)[:\s]*)?([Yy]es|[Nn]o)\b',
        # "Conclusion:\n...\nNo" - check conclusion section
        r'[Cc]onclusion[:\s]*\n?.*?([Yy]es|[Nn]o)\b',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_lower, re.MULTILINE | re.DOTALL)
        if match:
            return match.group(1).capitalize()
    
    # Check for numbered format in last few lines (e.g., "1. Yes" or "2. No")
    lines = response.strip().split('\n')
    for line in reversed(lines[-10:]):
        line_lower = line.strip().lower()
        # Match patterns like "1. Yes", "1) Yes", "1.yes", etc.
        if re.match(r'^\s*\d+[.\)]\s*yes\b', line_lower):
            return "Yes"
        if re.match(r'^\s*\d+[.\)]\s*no\b', line_lower):
            return "No"
    
    # Weak check: look for standalone yes/no in last 100 chars
    last_100 = response_lower[-100:]
    
    # Count occurrences (avoid "cannot", "does not", etc.)
    yes_match = re.search(r'\byes\b', last_100)
    no_match = re.search(r'\bno\b(?!\w)', last_100)  # "no" not followed by letter
    
    if yes_match and not no_match:
        return "Yes"
    elif no_match and not yes_match:
        return "No"
    
    # If both or neither found, check full end_text
    if "yes" in response_lower and "no" not in response_lower:
        return "Yes"
    elif "no" in response_lower and "yes" not in response_lower:
        return "No"
    
    logger.warning(f"Could not extract binary answer from response")
    return ""
