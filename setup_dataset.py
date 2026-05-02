import json
from datasets import load_dataset

def main():
    ds = load_dataset('reasoning-machines/gsm-hard', split='train')
    
    entries = []
    for i in range(100):
        entry = {
            "id": f"gsmhard_{i}",
            "query": ds[i]['input'],
            "expected_answer": float(ds[i]['target'])
        }
        entries.append(entry)
        
    with open('data/gsmhard_subset.json', 'w') as f:
        json.dump(entries, f, indent=2)
        
    print("First 3 entries:")
    print(json.dumps(entries[:3], indent=2))

if __name__ == "__main__":
    main()
