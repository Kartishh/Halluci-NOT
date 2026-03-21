from datasets import load_dataset
import json

ds = load_dataset("gsm8k", "main")

data = ds['test']

subset = []

for i in range(len(data)):
    q = data[i]["question"]
    ans_raw = data[i]["answer"]

    # simple heuristic: multi-step → longer + contains multiple operations
    if len(q.split()) > 20:
        answer = int(ans_raw.split("####")[-1].strip())

        subset.append({
            "query": q,
            "expected_output": answer
        })

    if len(subset) == 100:
        break

for i in range(100):
    question = data[i]["question"]
    answer_raw = data[i]["answer"]

    # Extract numeric answer
    answer = int(answer_raw.split("####")[-1].strip())

    subset.append({
        "query": question,
        "expected_output": answer
    })

with open("gsm_subset.json", "w") as f:
    json.dump(subset, f, indent=2)

print("✅ gsm_subset.json created successfully")