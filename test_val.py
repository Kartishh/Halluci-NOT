import re

def is_valid_expression(rhs):
    # Rule 3: remove things with consecutive operators like - -, + +, * *, / /, or missing operands
    # Check for consecutive operators (except for negative numbers like "* -5")
    # Actually, the user says if pattern like: "- -", "+ +", "* *", "/ /" -> FIX or discard line
    # Since they said fix or discard, let's just discard the entire line if it's structurally broken.
    if re.search(r'([+\-*/])\s+\1', rhs): # Matches "- -", "+ +", etc.
        return False
        
    # Strictly consecutive operators space-separated or not
    if re.search(r'[+*/]\s*[+*/]', rhs) or re.search(r'-\s*-', rhs) or re.search(r'[+*/]\s*-', rhs):
        # We can just be conservative and discard if it looks like broken math.
        # Let's simplify: any two operators in a row with or without spaces, EXCEPT a single - at the start of a number
        # Actually `3 * - 5` is weird anyway. 
        pass

    # Actually, the best way to check if an operand is missing:
    # Operators at the start (except an optional unary minus attached to a number)
    if re.match(r'^[+*/]', rhs):
        return False
    if re.match(r'^-\s+(?!\d|[a-z])', rhs): # "- " not followed by anything valid
        return False
        
    # Operator at the end
    if re.search(r'[+\-*/]\s*$', rhs):
        return False

    # Two operators in a row (e.g. "- -", "+ *")
    # A single space between operators or adjacent operators: "+ +", "- -", "* /"
    if re.search(r'[+\-*/]\s+[+\-*/]', rhs):
        return False
    # Adjacent operators like "++", "--", "**", "//"
    if re.search(r'[+\-*/]{2,}', rhs):
        return False

    # Also `x = + 5` operand is missing
    if re.match(r'^\+\s*\d', rhs):
        return False

    return True

print("5 * 3 :", is_valid_expression("5 * 3"))
print("- - 4 :", is_valid_expression("- - 4"))
print("16 - - 4 :", is_valid_expression("16 - - 4"))
print("+ 5 :", is_valid_expression("+ 5"))
print("5 * :", is_valid_expression("5 *"))
print("16 - 3 - 4 :", is_valid_expression("16 - 3 - 4"))
print("-5 :", is_valid_expression("-5"))
