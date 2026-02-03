# agents/refiner.py
from openai import OpenAI

client = OpenAI()

class Refiner:
    def __init__(self, model="gpt-4o"):
        self.model = model
        with open("prompts/reflexion.txt") as f:
            self.prompt = f.read()

    def reflect(self, claim, error_log):
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.prompt},
                {
                    "role": "user",
                    "content": f"CLAIM:\n{claim}\n\nERROR:\n{error_log}"
                }
            ],
            temperature=0
        )
        return response.choices[0].message.content
