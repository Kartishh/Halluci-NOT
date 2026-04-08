from symbolic.decomposer import AtomicFact
from verifier.pot_engine import get_pot_engine

facts = [
    AtomicFact("Assign", ["10", "percentage"], "", ""),
    AtomicFact("Assign", ["17", "percentage"], "", "")
]

pot = get_pot_engine().generate_script(facts)
print(pot.script)
