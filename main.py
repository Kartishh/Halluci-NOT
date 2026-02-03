# main.py
from agents.generator import Generator
from agents.decomposer import Decomposer
from agents.programmer import Programmer
from agents.refiner import Refiner
from environment.executor import Executor

gen = Generator()
dec = Decomposer()
prog = Programmer()
ref = Refiner()

def run(query, max_retries=3):
    claim = gen.invoke(query)

    for _ in range(max_retries):
        predicates = dec.decompose(claim)
        code = prog.generate_code(predicates)
        ok, result = Executor.run_code(code)

        if ok:
            return claim

        claim = ref.reflect(claim, result)

    return claim
