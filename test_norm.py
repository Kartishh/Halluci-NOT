import re

def fix_multi_equals(line):
    if '=' not in line:
        return [line]
    
    parts = line.split('=')
    lhs = parts[0].strip()
    results = []
    for var in parts[1:]:
        results.append(f"{lhs} = {var.strip()}")
    return results

print(fix_multi_equals("sellable_eggs = total_eggs - breakfast_eggs - muffin_eggs = 16 - 3 - 4 = 9"))
