
import os
import sys
import logging
from core.gemini_llm import get_gemini_llm
from core.reflexion import get_reflexion_engine

# Silence noisy loggers
logging.getLogger("LGP").setLevel(logging.INFO)

def reproduce():
    query = "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?"
    
    llm = get_gemini_llm()
    engine = get_reflexion_engine(llm)
    
    print("\n--- STARTING REPRODUCTION RUN ---\n")
    result = engine.run(query)
    print("\n--- END REPRODUCTION RUN ---\n")
    
    print(f"FINAL ANSWER: {result.final_answer}")
    print(f"BREADCRUMBS: {result.execution_trace}")

if __name__ == "__main__":
    reproduce()
