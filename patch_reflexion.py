import re

with open("core/reflexion.py", "r") as f:
    content = f.read()

# Replace the fallback block
old_fallback = """    # Fallback: simple redefinition check
    if not drifts:
        seen = {}
        for step_index, f in enumerate(facts):
            if f.predicate == "Assign" and len(f.arguments) == 2:
                val_str, var = f.arguments
                try:
                    val = float(val_str)
                except ValueError:
                    val = None

                if val is not None:
                    if var in seen and abs(seen[var] - val) > 1e-5:
                        if var not in formulas:
                            add_drift("redefinition_fallback", 0.75, var,
                                      seen[var], val,
                                      source_step=0, error_step=step_index)
                            break
                    seen[var] = val"""

new_fallback = """    # Task 3: Drift detection fallback when no formulas exist
    if not formulas:
        for var, history in value_history.items():
            if len(history) > 1:
                first = history[0]
                for curr in history[1:]:
                    if abs(curr["value"] - first["value"]) > 1e-5:
                        add_drift("redefinition", 0.8, var, first["value"], curr["value"], 
                                  source_step=first["step"], error_step=curr["step"])"""

if old_fallback in content:
    content = content.replace(old_fallback, new_fallback)
else:
    print("Warning: could not find old fallback block in reflexion.py")

with open("core/reflexion.py", "w") as f:
    f.write(content)

