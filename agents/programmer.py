# agents/programmer.py
class Programmer:
    def generate_code(self, predicates):
        lines = [
            "import math",
            "from environment.symbolic_table import SymbolicTable, SymbolicInconsistencyError",
            "",
            "sym = SymbolicTable()",
            "",
            "def verify():"
        ]

        indent = "    "

        for p in predicates:
            f = p["func"]
            args = p["args"]

            if f == "assign":
                var = p.get("var")
                val = args[0]
                lines.append(f"{indent}sym.assign('{var}', {val})")

            elif f == "multiply":
                a, b, res = args
                lines.append(
                    f"{indent}assert sym.get('{a}') * sym.get('{b}') == {res}"
                )

            elif f == "subtract":
                a, b, res = args
                lines.append(
                    f"{indent}assert sym.get('{a}') - {b} == {res}"
                )

            elif f == "percentage_increase":
                base, rate, res = args
                lines.append(
                    f"{indent}assert round(sym.get('{base}') * (1 + {rate}), 5) == {res}"
                )

        lines.append("")
        lines.append("verify()")

        return "\n".join(lines)
