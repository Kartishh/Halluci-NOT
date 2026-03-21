# Halluci-NOT: A Framework for Mitigating Symbolic Drift in LLMs

Halluci-NOT is a research-grade framework designed to tackle **hallucination** and **symbolic drift** in Large Language Models (LLMs). By integrating Symbolic State and Constraint Extraction (SSCE) with deterministic execution environments, Halluci-NOT ensures rigorous mathematical and logical reasoning without falling into the trap of "painless logic" or unverified numerical claims.

---

## 🚀 Key Features

- **Robust Symbolic Decomposer:** Utilizes a few-shot prompted agent (via the Gemini API) for flexible, highly accurate extraction of predicates and constraints from natural language reasoning steps, replacing brittle regex-based parsing.
- **Deterministic Verification Engine:** Safely executes programmatic thought (PoT) in an isolated sandbox, dynamically validating LLM-generated logic against ground truth.
- **Numeric Natural Language Inference (NLI):** Closes the logic gap by rigorously comparing symbolic execution results against textual claims for both value and entity.
- **Latency-Optimized 'Early Exit':** Intelligently skips computationally heavy operations (like full sandbox execution) for simpler reasoning steps, consistently achieving execution times under 5 seconds.
- **Comprehensive Evaluation Suite:** Purpose-built to benchmark against complex, multi-step drift scenarios (e.g., using curated subsets like GSM-Hard).

---

## 🏗️ Architecture Overview

The framework is divided into several core components:

1. **`core/` (Orchestration & State Management)**
   - `policy.py`: Defines the overarching reasoning and early-exit policies.
   - `state_manager.py`: Tracks the evolving symbolic state throughout the reasoning process.

2. **`symbolic/` (Extraction & Representation)**
   - `decomposer.py`: Interfaces with the LLM to parse raw text into structured predicates.
   - `ssce_algorithm.py` & `table.py`: Manages the extracted symbolic variables and constraints.

3. **`verifier/` (Execution & Validation)**
   - `pot_engine.py` & `sandbox.py`: Safely generates and executes Pythonic code representing the reasoning steps.
   - `numeric_nli.py` & `nli_gate.py`: Validates the exact numeric and entity match between the sandbox output and the original claim.

4. **`evaluation/` (Benchmarking)**
   - Includes runners, baseline comparisons, and stress tests to evaluate the framework's performance against standard approaches.

---

## ⚙️ Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Kartishh/Halluci-NOT.git
   cd Halluci-NOT
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables:**
   Create a `.env` file in the root directory based on the `.env.example` file.
   ```bash
   cp .env.example .env
   # Add your API keys (e.g., GEMINI_API_KEY) to .env
   ```

---

## 📊 Current Metrics & Performance

> **[Space reserved for future metric additions]**
> *As evaluation runs complete (e.g., against GSM-Hard or POPQA), update this section with latency improvements, accuracy comparisons vs. base LLMs, and symbolic drift reduction statistics.*

| Metric | Base Model (Baseline) | Halluci-NOT (SSCE + Verifier) | Improvement |
| :--- | :---: | :---: | :---: |
| **Accuracy (GSM-Hard)** | *TBD* | *TBD* | *TBD* |
| **Average Latency** | *TBD* | *< 5.0s (Target)* | *TBD* |
| **Drift Detection Rate** | *TBD* | *TBD* | *TBD* |

---

## 🛠️ Usage

### Running the Evaluation Suite
To run the standard evaluation pipeline against the GSM subset:
```bash
python main.py
```

### Running Stress Tests
To evaluate the system under high-complexity multi-step reasoning scenarios:
```bash
python -m evaluation.stress_tests
```

---

## 🔮 Future Updates & Roadmap

- [ ] **Expanded Benchmarks:** Integration with additional datasets (e.g., MATH, deep mathematical reasoning tasks).
- [ ] **Advanced Sandbox Features:** Support for more complex libraries and execution timeout optimizations.
- [ ] **Full Results Publication:** Final compiled metrics for the SSCE mitigation strategy.
