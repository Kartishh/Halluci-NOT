from symbolic.decomposer import normalize_reasoning

text1 = """
sellable_eggs = total_eggs - breakfast_eggs - muffin_eggs
sellable_eggs = 16 - 3 - 4
"""

text2 = "sellable_eggs = total_eggs - breakfast_eggs - muffin_eggs = 16 - 3 - 4 = 9"
text3 = "x = a + b"

print("--- TEST 1: Preserve multiple definition lines ---")
print(normalize_reasoning(text1).strip())

print("\n--- TEST 2: Expand chained equalities ---")
print(normalize_reasoning(text2).strip())

print("\n--- TEST 3: Preserve single-letter variable names ---")
print(normalize_reasoning(text3).strip())
