from core.gemini_llm import get_gemini_llm
from core.reflexion import get_reflexion_engine

print("Running test...")

llm = get_gemini_llm()
engine = get_reflexion_engine(llm)

query = "If x = 5 and y = 3, what is x * y?"
forced_reasoning = "x = 5\ny = 3\nresult = x * y\nFINAL ANSWER: 15"

res = engine.run(query, forced_reasoning=forced_reasoning)
print("Result final answer:", res.final_answer)
print("Result drift detected:", res.drift_detected)
