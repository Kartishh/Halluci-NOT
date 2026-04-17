import re
from symbolic.decomposer import _UNIT_WORDS, _TABLE_PATTERNS

# [I will mock out the dependencies for local testing]
def _expand_double_expressions(line: str) -> list:
    if '=' not in line:
        return [line]
    parts = line.split('=')
    lhs = parts[0].strip()
    if len(parts) <= 2:
        return [line]
    results = []
    for rhs in parts[1:]:
        results.append(f"{lhs} = {rhs.strip()}")
    # Filter out single-constant final assignment IF there's a previous computation
    # The user said: "NEVER collapse expressions into constants... KEEP variable-based expressions first"
    # Actually, returning all forms is safe because `pot_engine` sorts and filters them later.
    return results

def normalize_reasoning(text: str) -> str:
    if not text:
        return text

    # Basic cleaning
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'(?<!\w)\*([^*]+)\*(?!\w)', r'\1', text)
    text = re.sub(r'^\s*[\*\-•]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*(?:step\s+)?\d+[\.)\:\s]+\s*', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^#+\s+.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*(?:step[- ]by[- ]step\s+)?(?:solution|explanation|reasoning|answer)\s*[:.]?\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)

    # LaTeX
    text = re.sub(r'\\\[\s*', '\n', text)
    text = re.sub(r'\s*\\\]', '\n', text)
    text = re.sub(r'\\\(\s*', '', text)
    text = re.sub(r'\s*\\\)', '', text)
    text = text.replace('\\_', '_')
    text = re.sub(r'\\text\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\textbf\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\textit\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1/\2)', text)
    text = re.sub(r'\\times\b', ' * ', text)
    text = re.sub(r'\\cdot\b', ' * ', text)
    text = re.sub(r'\\div\b', ' / ', text)
    text = re.sub(r'\\boxed\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\(?:left|right)[(\[{)\]}.]?', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = text.replace('{', '').replace('}', '')
    text = text.replace('[', '').replace(']', '')
    text = re.sub(r'\\+', '', text)

    # Unit words note: Not applied globally anymore.
    text = text.lower()

    # Noise removal (Fixed: removed single letter matches like a, an, of, in that destroy vars)
    text = re.sub(
        r'\b(?:is|are|was|were|equals?|equal\s+to|which\s+is|that\s+is|'
        r'gives?|giving|we\s+get|we\s+have|we\s+find|this\s+means|'
        r'so\s+that|therefore|thus|hence|then)\b', '', text)

    # Dollar and syntax
    text = re.sub(r'\$\s*', '', text)
    text = re.sub(r'(\d),(?=\d{3})', r'\1', text)
    text = re.sub(r'(\d)\.\s', r'\1 ', text)

    lines = text.split('\n')
    relevant = []
    for line in lines:
        line = line.strip()
        if not line: continue
        if _TABLE_PATTERNS.match(line): continue
        if '=' in line:
            relevant.append(line)
            continue
        if re.search(r'[\+\-\*/]', line) and re.search(r'\d', line):
            relevant.append(line)
            continue
        if len(line) < 40 and re.match(r'^[\w\s\+\-\*/\(\)\.\d_]+$', line):
            if re.search(r'\d', line):
                relevant.append(line)

    expanded = []
    for line in relevant:
        expanded.extend(_expand_double_expressions(line))
    relevant = expanded

    # Operator reconstruction not mocked here, we assume it's correct.
    
    return '\n'.join(relevant)

print("--- TEST 1: Double Expressions ---")
print(normalize_reasoning("sellable_eggs = total_eggs - breakfast_eggs - muffin_eggs = 16 - 3 - 4 = 9"))

print("\n--- TEST 2: Multi-line inputs ---")
print(normalize_reasoning("x = a + b\nx = 3 + 5"))
