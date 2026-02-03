# environment/symbolic_table.py

class SymbolicInconsistencyError(Exception):
    pass


class SymbolicTable:
    """
    Tracks symbolic variable assignments across sub-claims.
    Detects logical inconsistencies like variable redefinition.
    """

    def __init__(self):
        self.table = {}

    def assign(self, var_name, value):
        if var_name in self.table:
            if self.table[var_name] != value:
                raise SymbolicInconsistencyError(
                    f"Variable '{var_name}' was previously "
                    f"{self.table[var_name]} but reassigned to {value}"
                )
        self.table[var_name] = value

    def get(self, var_name):
        if var_name not in self.table:
            raise SymbolicInconsistencyError(
                f"Variable '{var_name}' used before assignment"
            )
        return self.table[var_name]

    def snapshot(self):
        return dict(self.table)
