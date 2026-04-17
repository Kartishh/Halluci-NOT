"""
core/pot_converter.py

HalluciNOT (LGP)
---------------------------------
Program-of-Thought (PoT) Converter

Purpose:
    Convert natural language reasoning into structured pseudo-code lines
    that the regex-based decomposer can parse deterministically.

Pipeline position:
    LLM reasoning → PoT conversion → symbolic decomposition

Design:
    - Extract variable assignments from NL
    - Convert word-numbers to numeric form
    - Normalize variable names (snake_case)
    - Only generate simple arithmetic expressions
    - Ignore irrelevant/narrative text

Author: LGP Framework
"""

from __future__ import annotations

import re
import logging
from typing import List, Optional, Tuple, Dict

logger = logging.getLogger("LGP.PoTConverter")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Word-to-Number Mapping
# ---------------------------------------------------------------------------

_WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
    "million": 1_000_000, "half": 0.5, "quarter": 0.25,
    "twice": 2, "double": 2, "triple": 3,
}

# Operator keyword mapping (NL → symbol)
_OP_KEYWORDS = {
    "plus": "+", "added to": "+", "combined with": "+",
    "add": "+", "increased by": "+", "more than": "+",
    "minus": "-", "subtracted": "-", "less than": "-",
    "subtract": "-", "decreased by": "-", "reduced by": "-",
    "times": "*", "multiplied by": "*", "multiplied": "*",
    "multiply": "*", "product of": "*",
    "divided by": "/", "divided": "/", "over": "/",
    "divide": "/", "per": "/",
}


# ---------------------------------------------------------------------------
# Variable Name Normalization
# ---------------------------------------------------------------------------

def _canonicalize(name: str) -> str:
    """Remove underscores and spaces, convert to lowercase. PRESERVED for compatibility."""
    return name.replace('_', '').replace(' ', '').lower()


def _normalize_var_name(name: str) -> str:
    """Convert a natural language entity name to snake_case. PRESERVES variable identity (including digits)."""
    original = name.strip()
    name = original.lower()
    
    name = re.sub(r'^(?:step\s+)?\d+[\.):\s]+\s*', '', name, flags=re.IGNORECASE)
    name = re.sub(r"'s\b", "", name)
    # Do NOT remove digits - preserve variable identity like p1, p2, x2, etc.
    name = re.sub(r"[^a-z0-9\s]", "", name)  # Keep digits!
    name = re.sub(r"\s+", "_", name.strip())
    name = name.strip("_")
    name = re.sub(r"_+", "_", name)
    parts = name.split('_')
    if len(parts) > 3:
        name = '_'.join(parts[:3])
    # Preserve single letter/digit variables
    if not name and original:
        return original
    return name if name else ""


def _word_to_number(word: str) -> Optional[float]:
    """Convert a word-number to its numeric value. Returns None if not a number."""
    word = word.strip().lower().replace(",", "")
    # Direct numeric
    try:
        return float(word)
    except ValueError:
        pass
    # Currency prefix
    if word.startswith("$"):
        try:
            return float(word[1:].replace(",", ""))
        except ValueError:
            pass
    # Word lookup
    if word in _WORD_NUMBERS:
        return float(_WORD_NUMBERS[word])
    return None


def _extract_number(text: str) -> Optional[str]:
    """Extract a numeric value from text, handling commas and currency."""
    text = text.strip()
    # Try $NUMBER or NUMBER
    m = re.search(r"\$?\s*([-+]?\d[\d,]*\.?\d*)", text)
    if m:
        return m.group(1).replace(",", "")
    # Try word numbers
    for word, val in _WORD_NUMBERS.items():
        if word in text.lower():
            return str(val)
    return None


# ---------------------------------------------------------------------------
# Sentence-Level Converters
# ---------------------------------------------------------------------------

def _try_explicit_assignment(sentence: str) -> Optional[str]:
    """
    Match: "X = NUMBER" or "X = VARIABLE" or "X = NUMBER with extra text"
    Output: "x = NUMBER" or "x = variable"
    Extracts first numeric value from RHS, ignores trailing words.
    REJECTS if multiple numbers present (e.g., "60 miles in 2 hours").
    """
    s = sentence.strip()
    m = re.match(
        r"(?:let\s+)?(\w[\w\s]*?)\s*=\s*(.+)$",
        s, re.IGNORECASE
    )
    if m:
        var = _normalize_var_name(m.group(1))
        if not var:
            return None
        rhs = m.group(2).strip()
        numbers = re.findall(r'\d[\d,]*\.?\d*', rhs)
        
        # Allow: either exactly one number OR no numbers (variable = variable)
        if len(numbers) > 1:
            return None
        
        if len(numbers) == 1:
            num_match = re.search(r'([-+]?\$?\d[\d,]*\.?\d*)', rhs)
            if num_match:
                val = num_match.group(1).replace(",", "").replace("$", "")
                return f"{var} = {val}"
        else:
            # No numbers - variable = variable case
            # Check if RHS is a valid variable name
            rhs_var = rhs.strip()
            if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', rhs_var):
                return f"{var} = {rhs_var}"
    return None


def _try_explicit_arithmetic(sentence: str) -> Optional[str]:
    """
    Match: "X = A + B" / "X = A * B" etc.
    Handles commas in numbers (e.g. 80,000 + 50,000)
    and decimals (e.g. 60 * 2.5).
    Output: "x = a + b" (with resolved variable names)
    """
    # Allow commas and dots in numeric operands
    m = re.match(
        r"(?:let\s+)?(\w[\w\s]*?)\s*=\s*([\w.,\s$]*?)\s*([+\-*/])\s*([\w.,\s$]*?)\.?\s*$",
        sentence.strip(), re.IGNORECASE
    )
    if m:
        result = _normalize_var_name(m.group(1))
        arg1_raw = m.group(2).strip()
        op = m.group(3)
        arg2_raw = m.group(4).strip()

        # Resolve args: could be numbers (with commas/$) or variable names
        arg1_clean = arg1_raw.replace(",", "").replace("$", "")
        arg2_clean = arg2_raw.replace(",", "").replace("$", "")

        arg1_num = _word_to_number(arg1_clean)
        arg2_num = _word_to_number(arg2_clean)

        arg1 = str(arg1_num) if arg1_num is not None else _normalize_var_name(arg1_raw)
        arg2 = str(arg2_num) if arg2_num is not None else _normalize_var_name(arg2_raw)

        # Convert int-like floats: 80000.0 → 80000
        for val_ref in ["arg1", "arg2"]:
            val = locals()[val_ref]
            try:
                fv = float(val)
                if fv == int(fv):
                    locals()[val_ref]
                    if val_ref == "arg1":
                        arg1 = str(int(fv))
                    else:
                        arg2 = str(int(fv))
            except ValueError:
                pass

        return f"{result} = {arg1} {op} {arg2}"
    return None


def _try_nl_assignment(sentence: str) -> Optional[str]:
    """
    Match: "The ENTITY is/costs/equals NUMBER"
    Output: "entity = NUMBER"
    Extracts first numeric value, ignores trailing words.
    REJECTS if multiple numbers present in sentence.
    """
    if re.search(r'\b(has|have|had)\s+\d+\s+times\b', sentence.lower()):
        return None
    
    all_numbers = re.findall(r'\d[\d,]*\.?\d*', sentence)
    if len(all_numbers) != 1:
        return None
        
    patterns = [
        r"(?:the\s+)?(.+?)\s+(?:is|are|was|were|has|have|had|costs?|equals?|=)\s+\$?([-+]?\d[\d,]*\.?\d*)",
        r"(?:he|she|it|they|there)\s+(?:has|have|had|earns?|makes?|gets?|pays?)\s+\$?([-+]?\d[\d,]*\.?\d*)",
        r"^([-+]?\$?\d[\d,]*\.?\d*)\s+([\w\s]+?)(?:\.|$)",
    ]

    for pat in patterns[:2]:
        m = re.search(pat, sentence, re.IGNORECASE)
        if m:
            entity = m.group(1)
            val = m.group(2).replace(",", "").replace("$", "")
            var = _normalize_var_name(entity)
            if var and len(var) > 1:
                try:
                    float(val)
                    return f"{var} = {val}"
                except ValueError:
                    pass

    m = re.search(patterns[2], sentence, re.IGNORECASE)
    if m:
        val = m.group(1).replace(",", "").replace("$", "")
        entity = m.group(2)
        var = _normalize_var_name(entity)
        if var and len(var) > 1:
            try:
                float(val)
                return f"{var} = {val}"
            except ValueError:
                pass

    for word, num in _WORD_NUMBERS.items():
        pat = rf"\b{word}\b"
        if re.search(pat, sentence, re.IGNORECASE):
            m = re.search(
                rf"(.+?)\s+(?:is|are|was|were|costs?|equals?)\s+{word}\b",
                sentence, re.IGNORECASE
            )
            if m:
                var = _normalize_var_name(m.group(1))
                if var and len(var) > 1:
                    return f"{var} = {num}"

    return None
    patterns = [
        r"(?:the\s+)?(.+?)\s+(?:is|are|was|were|has|have|had|costs?|equals?|=)\s+\$?([-+]?\d[\d,]*\.?\d*)",
        r"(?:he|she|it|they|there)\s+(?:has|have|had|earns?|makes?|gets?|pays?)\s+\$?([-+]?\d[\d,]*\.?\d*)",
        r"^([-+]?\$?\d[\d,]*\.?\d*)\s+([\w\s]+?)(?:\.|$)",
    ]

    for pat in patterns[:2]:
        m = re.search(pat, sentence, re.IGNORECASE)
        if m:
            entity = m.group(1)
            val = m.group(2).replace(",", "").replace("$", "")
            var = _normalize_var_name(entity)
            if var and len(var) > 1:
                try:
                    float(val)
                    return f"{var} = {val}"
                except ValueError:
                    pass

    m = re.search(patterns[2], sentence, re.IGNORECASE)
    if m:
        val = m.group(1).replace(",", "").replace("$", "")
        entity = m.group(2)
        var = _normalize_var_name(entity)
        if var and len(var) > 1:
            try:
                float(val)
                return f"{var} = {val}"
            except ValueError:
                pass

    for word, num in _WORD_NUMBERS.items():
        pat = rf"\b{word}\b"
        if re.search(pat, sentence, re.IGNORECASE):
            m = re.search(
                rf"(.+?)\s+(?:is|are|was|were|costs?|equals?)\s+{word}\b",
                sentence, re.IGNORECASE
            )
            if m:
                var = _normalize_var_name(m.group(1))
                if var and len(var) > 1:
                    return f"{var} = {num}"

    return None


def _try_nl_arithmetic(sentence: str) -> Optional[str]:
    """
    Match NL arithmetic patterns:
      "total is cost plus tax"
      "profit is revenue minus expenses"
      "area is length times width"
    Output: structured "result = a OP b"
    """
    for keyword, op_symbol in _OP_KEYWORDS.items():
        # Pattern: "RESULT is/= A KEYWORD B"
        pat = rf"(?:the\s+)?(.+?)\s+(?:is|are|=|equals?)\s+(?:the\s+)?(.+?)\s+{keyword}\s+(?:the\s+)?(.+?)(?:\.|$)"
        m = re.search(pat, sentence, re.IGNORECASE)
        if m:
            result = _normalize_var_name(m.group(1))
            arg1_raw = m.group(2).strip()
            arg2_raw = m.group(3).strip()

            arg1_num = _word_to_number(arg1_raw)
            arg2_num = _word_to_number(arg2_raw)

            arg1 = str(arg1_num) if arg1_num is not None else _normalize_var_name(arg1_raw)
            arg2 = str(arg2_num) if arg2_num is not None else _normalize_var_name(arg2_raw)

            if result and arg1 and arg2:
                return f"{result} = {arg1} {op_symbol} {arg2}"

    # Pattern: "A KEYWORD B gives/yields RESULT"
    for keyword, op_symbol in _OP_KEYWORDS.items():
        pat = rf"(?:the\s+)?(.+?)\s+{keyword}\s+(?:the\s+)?(.+?)\s+(?:gives?|yields?|equals?|is)\s+(?:the\s+)?(.+?)(?:\.|$)"
        m = re.search(pat, sentence, re.IGNORECASE)
        if m:
            arg1_raw = m.group(1).strip()
            arg2_raw = m.group(2).strip()
            result = _normalize_var_name(m.group(3))

            arg1_num = _word_to_number(arg1_raw)
            arg2_num = _word_to_number(arg2_raw)

            arg1 = str(arg1_num) if arg1_num is not None else _normalize_var_name(arg1_raw)
            arg2 = str(arg2_num) if arg2_num is not None else _normalize_var_name(arg2_raw)

            if result and arg1 and arg2:
                return f"{result} = {arg1} {op_symbol} {arg2}"

    return None


def _try_percentage_assignment(sentence: str) -> Optional[str]:
    """
    Match: "X percent" / "X%" patterns and convert to decimal.
    Output: "variable = decimal_value"
    """
    # "tax is 8%" / "discount is 20 percent"
    m = re.search(
        r"(?:the\s+)?(\w[\w\s]*?)\s+(?:is|are|=)\s+([-+]?\d[\d,]*\.?\d*)\s*(?:%|percent)",
        sentence, re.IGNORECASE
    )
    if m:
        var = _normalize_var_name(m.group(1))
        pct = float(m.group(2).replace(",", ""))
        return f"{var} = {pct / 100}"
    return None


# ---------------------------------------------------------------------------
# Strict Filtering (Structural Validation Only)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Unit and Noise Removal
# ---------------------------------------------------------------------------

_UNIT_WORDS = {
    'mph', 'kmh', 'km', 'm', 'cm', 'mm',
    'mile', 'miles', 'mi',
    'hour', 'hours', 'hr', 'h',
    'minute', 'minutes', 'min',
    'second', 'seconds', 'sec',
    'kg', 'g', 'mg',
    'liter', 'liters', 'l',
    'dollar', 'dollars', 'usd',
    'percent', 'pct',
}

_STRIP_CHARS = r'[;`→|\\$]+'

_LET_PREFIX = re.compile(r'^let\s+', re.IGNORECASE)


def _preprocess_line(line: str) -> str:
    """Remove units, stray symbols, and 'let' prefix from a line."""
    line = line.strip()
    
    line = _LET_PREFIX.sub('', line)
    
    line = re.sub(_STRIP_CHARS, '', line)
    
    tokens = line.split()
    cleaned_tokens = []
    for token in tokens:
        if token.lower() not in _UNIT_WORDS:
            cleaned_tokens.append(token)
        else:
            pass
    
    result = ' '.join(cleaned_tokens)
    
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result


def _strict_filter(line: str) -> str | None:
    """
    Apply strict structural rules to validate a single assignment line.
    Returns the valid line or None if it fails any rule.
    
    PRESERVES exact variable names - no canonicalization.
    """
    if not line or "=" not in line:
        return None

    var_name, rhs = line.split("=", 1)
    var_name = var_name.strip()
    rhs = rhs.strip()

    if not var_name or not rhs:
        return None

    # Validate variable name - preserve EXACT case and characters
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', var_name):
        return None

    has_operator = bool(re.search(r'[+\-*/()]', rhs))

    if has_operator:
        # Validate RHS - preserve variable names exactly
        if not re.match(r'^[0-9a-zA-Z_+\-*/().\s]+$', rhs):
            return None
        has_number = bool(re.search(r'\d', rhs))
        if not has_number and not has_operator:
            return None
        # Preserve exact variable names in expression
        return f"{var_name} = {rhs}"
    else:
        # Allow: variable = number OR variable = variable
        numbers = re.findall(r'\d[\d,]*\.?\d*', rhs)
        variables = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', rhs)
        
        # Must have either numbers or valid variables
        if len(numbers) > 1:
            return None
        if not numbers and not variables:
            return None
            
        if numbers:
            num_match = re.search(r'([-+]?\$?\d[\d,]*\.?\d*)', rhs)
            if not num_match:
                return None
            val = num_match.group(1).replace(",", "").replace("$", "")
            return f"{var_name} = {val}"
        else:
            # variable = variable case - preserve exact variable name
            return f"{var_name} = {rhs}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sanitize_variable(var: str) -> str:
    """
    Sanitize variable names to be safe and robust.
    """
    var = var.lower().strip()

    # Keep underscores, remove only invalid chars
    var = re.sub(r'[^a-z0-9_]', '_', var)

    # Collapse multiple underscores
    var = re.sub(r'_+', '_', var).strip('_')

    # FIX: ensure valid Python identifier
    if not var or var[0].isdigit():
        var = f"v_{var}"

    # Avoid empty or generic names
    if var in ["", "var", "value"]:
        return "INVALID"

    return var[:30]


def _preprocess_sentence(sentence: str) -> str:
    """Normalize a sentence before conversion: Unicode operators, step prefixes, chained equals, unit removal."""
    s = sentence.strip()
    
    s = re.sub(r'^let\s+', '', s, flags=re.IGNORECASE)
    
    s = re.sub(r'[;`→|\\$]+', '', s)
    
    tokens = s.split()
    cleaned_tokens = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        token_lower = token.lower()
        
        if token == '+' or token == '-' or token == '*' or token == '/':
            cleaned_tokens.append(token)
        elif token_lower in _UNIT_WORDS:
            pass
        else:
            cleaned_tokens.append(token)
        i += 1
    
    s = ' '.join(cleaned_tokens)
    
    s = re.sub(r'\s+', ' ', s).strip()
    
    s = s.replace('\u00d7', '*').replace('\u00f7', '/').replace('\u2212', '-')
    s = s.replace('\u2013', '-').replace('\u2014', '-')
    s = re.sub(r'^(?:step\s+)?\d+[\.):\s]+\s*', '', s, flags=re.IGNORECASE)
    equals_positions = [m.start() for m in re.finditer(r'\s*=\s*', s)]
    if len(equals_positions) >= 2:
        s = s[:equals_positions[1]].strip()
    return s


def reasoning_to_program(reasoning: str) -> List[str]:
    """
    Convert natural language reasoning into structured pseudo-code.

    Each line of output is a simple assignment or arithmetic expression
    that the regex-based decomposer can parse deterministically.

    Args:
        reasoning: Natural language reasoning text from LLM

    Returns:
        List of pseudo-code strings, e.g.
        ["eggs_per_day = 16", "eggs_eaten = 3", "remaining = eggs_per_day - eggs_eaten"]

    Rules:
        - Extract variable assignments
        - Convert numbers to numeric form (no commas, no $)
        - Normalize variable names to snake_case
        - Only generate simple arithmetic expressions (=, +, -, *, /)
        - Ignore irrelevant narrative text
    """
    if not reasoning or not reasoning.strip():
        return []

    # Split reasoning into sentences/steps (include comma before capital letter)
    sentences = re.split(r'(?:\. |\n|; |\.\n|,\s+(?=[A-Z]))', reasoning)
    sentences = [s.strip().rstrip(".") for s in sentences if s.strip()]

    valid_lines: List[str] = []

    for sentence in sentences:
        # Preprocess: Unicode operators, step prefixes, chained equals
        sentence = _preprocess_sentence(sentence)
        if not sentence:
            continue

        line = None

        # Try converters in order of specificity
        line = _try_explicit_arithmetic(sentence)
        if line is None:
            line = _try_explicit_assignment(sentence)
        if line is None:
            line = _try_percentage_assignment(sentence)
        if line is None:
            line = _try_nl_arithmetic(sentence)
        if line is None:
            line = _try_nl_assignment(sentence)

        if line is not None:
            filtered = _strict_filter(line)
            if filtered is not None:
                valid_lines.append(filtered)

    if len(valid_lines) > 15:
        valid_lines = valid_lines[-15:]

    if valid_lines:
        logger.info(
            f"PoT converter: {len(valid_lines)} lines from "
            f"{len(sentences)} sentences"
        )

    return valid_lines


# ---------------------------------------------------------------------------
# Multi-Pass Decomposition Helper
# ---------------------------------------------------------------------------

def multi_pass_validate(program_lines: List[str]) -> List[str]:
    """
    Multi-pass validation of program lines:
        Pass 1: Extract all defined variables
        Pass 2: Extract all operations and their inputs
        Pass 3: Validate that all operation inputs are defined

    Returns the validated (and potentially filtered) program lines.
    Logs warnings for undefined variable usage.
    """
    defined_vars: set = set()
    operations: List[Tuple[int, str, List[str]]] = []  # (index, result_var, input_vars)

    # --- Pass 1: Extract variable definitions ---
    for i, line in enumerate(program_lines):
        if "=" not in line:
            continue
        parts = line.split("=", 1)
        var_name = parts[0].strip()
        defined_vars.add(var_name)

    # --- Pass 2: Extract operations and their inputs ---
    for i, line in enumerate(program_lines):
        if "=" not in line:
            continue
        parts = line.split("=", 1)
        result_var = parts[0].strip()
        rhs = parts[1].strip()

        # Check if RHS has arithmetic operators
        for op in ["+", "-", "*", "/"]:
            if op in rhs:
                operands = [o.strip() for o in re.split(r"[+\-*/]", rhs)]
                input_vars = []
                for operand in operands:
                    try:
                        float(operand)
                    except ValueError:
                        input_vars.append(operand)
                operations.append((i, result_var, input_vars))
                break

    # --- Pass 3: Validate dependencies ---
    validated_lines = list(program_lines)
    for idx, result_var, input_vars in operations:
        for inp in input_vars:
            if inp and inp not in defined_vars:
                logger.warning(
                    f"Multi-pass validation: variable '{inp}' used in "
                    f"'{result_var}' but not defined. Line: {program_lines[idx]}"
                )

    return validated_lines


# ---------------------------------------------------------------------------
# Convenience: decompose_line (for integration with decomposer)
# ---------------------------------------------------------------------------

def decompose_line(line: str):
    """
    Convert a single program line into AtomicFact(s).

    This is a thin wrapper that delegates to the decomposer's
    regex path. Imported lazily to avoid circular imports.
    """
    from symbolic.decomposer import SymbolicDecomposer
    decomposer = SymbolicDecomposer()
    return decomposer._rule_based_extract(line)
