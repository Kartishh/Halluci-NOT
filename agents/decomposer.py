# agents/decomposer.py
import json
from utils.parser import extract_json
from openai import OpenAI

client = OpenAI()

class Decomposer:
    def __init__(self, model="gpt-4o"):
        self.model = model
        with open("prompts/decomposition.txt") as f:
            self.system_prompt = f.read()

    def decompose(self, text: str):
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0
        )

        raw = response.choices[0].message.content
        return extract_json(raw)
