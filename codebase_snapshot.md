# Codebase Snapshot

## 1. FULL FILE TREE

```text
.
./append_report.py
./Bibliography - Actual used
./Bibliography - Actual used/[10]When_not_to_trust_LLM.pdf
./Bibliography - Actual used/[11]Check_your_facts.pdf
./Bibliography - Actual used/[12]RARR.pdf
./Bibliography - Actual used/[13]Self_Check_Agent.pdf
./Bibliography - Actual used/[14]REACT.pdf
./Bibliography - Actual used/[15]Toolformer.pdf
./Bibliography - Actual used/[16]Chain_of_Thought_Prompting.pdf
./Bibliography - Actual used/[17]Zero-shot_Reasoners.pdf
./Bibliography - Actual used/[18]Self_Refine.pdf
./Bibliography - Actual used/[19]Solve_Computer_Tasks.pdf
./Bibliography - Actual used/[1]Pelican.pdf
./Bibliography - Actual used/[20]Step_by_step.pdf
./Bibliography - Actual used/[21]LLM_as_a_judge.pdf
./Bibliography - Actual used/[22]OpenAI_GPT4.pdf
./Bibliography - Actual used/[23]Llama2.pdf
./Bibliography - Actual used/[24]TRUE_Factual_Consistency.pdf
./Bibliography - Actual used/[25]QA_with_LM.pdf
./Bibliography - Actual used/[26]Peogram_of_Thoughts.pdf
./Bibliography - Actual used/[2]Reflexion.pdf
./Bibliography - Actual used/[3]Chain_of_Verification_Thought.pdf
./Bibliography - Actual used/[4]FACTOOL.pdf
./Bibliography - Actual used/[5]HaluEval.pdf
./Bibliography - Actual used/[6]Zhang_et_al_Snowball.pdf
./Bibliography - Actual used/[7]Self-RAG.pdf
./Bibliography - Actual used/[8]Factscore.pdf
./Bibliography - Actual used/[9]Hallucination_in_LLM.pdf
./core
./core/gemini_llm.py
./core/__init__.py
./core/nvidia_llm.py
./core/policy.py
./core/pot_converter.py
./core/__pycache__
./core/reflexion.py
./core/state_manager.py
./data
./data/controlled_drift_dataset.json
./data/drift_stress_test.json
./data/gsm8k_subset.json
./data/gsmhard_subset.json
./data/__init__.py
./data/logger.py
./data/mawps_subset.json
./data/popqa_loader.py
./data/__pycache__
./data/svamp_subset.json
./Dockerfile
./.env
./.env.example
./evaluation
./evaluation/baselines.py
./evaluation/datasets.py
./evaluation/export.py
./evaluation/__init__.py
./evaluation/liar_agent.py
./evaluation/metrics.py
./evaluation/__pycache__
./evaluation/runner.py
./evaluation/stress_tests.py
./final_benchmark_eval.py
./generate_fresh_results.py
./generate_full_report.py
./generate_report.py
./generate_snapshot.py
./.git
./.gitignore
./gsm_dataset.py
./gsm_subset.json
./lgp_eval.py
./main.py
./patch_reflexion.py
./project_report.md
./__pycache__
./README.md
./real_eval.py
./reproduce_gsm_03.py
./requirements.txt
./research_eval.py
./results
./results/accuracy_comparison.png
./results/analysis_report.txt
./results/correction_success_rate.png
./results/debug_trace_gsm_03.txt
./results/demo_trace.txt
./results/drift_detection_rate.png
./results/ensemble_ablation_table.txt
./results/evaluation_results.csv
./results/factored_eval_metrics.json
./results/final_benchmark_metrics.json
./results/final_comparison_table.txt
./results/final_demo_trace.txt
./results/final_metrics.json
./results/final_metrics_table.txt
./results/final_presentation_tables.txt
./results/final_real_eval_table.txt
./results/final_report_tables.md
./results/improvement_vs_regression.png
./results/lgp_evaluation_report.md
./results/lgp_evaluation_results.csv
./results/lgp_summary_metrics.json
./results/mawps_real_eval_table.txt
./results/mawps_summary_metrics.json
./results/plots
./results/plots/accuracy_comparison.png
./results/plots/before_vs_after.png
./results/plots/drift_detection.png
./results/plots/drift_type_pie.png
./results/plots/error_breakdown.png
./results/plots/improvement_vs_regression.png
./results/plots/logic_type_breakdown.png
./results/pot_validation_results.json
./results/progress_ensemble.json
./results/progress_eval.json
./results/progress_mawps.json
./results/progress_svamp.json
./results/research_summary.txt
./results/results_after.csv
./results/results_before.csv
./results/summary_metrics_before.json
./results/summary_metrics.json
./results/summary_report.txt
./results/svamp_real_eval_table.txt
./results/svamp_summary_metrics.json
./results/temp_progress.json
./run_factored_eval.py
./run_final_eval.py
./scratch
./scratch/generate_svamp.py
./scratch/inspect_svamp.py
./setup_dataset.py
./summary_metrics.json
./symbolic
./symbolic/decomposer.py
./symbolic/__init__.py
./symbolic/__pycache__
./symbolic/ssce_algorithm.py
./symbolic/table.py
./synthetic_drift.json
./temp_real_eval.py
./validate_pot_upgrade.py
./verifier
./verifier/factored_verifier.py
./verifier/__init__.py
./verifier/nli_gate.py
./verifier/numeric_nli.py
./verifier/pot_engine.py
./verifier/__pycache__
./verifier/sandbox.py
```

## 2. ENTRY POINTS

Found main entry files:
- main.py
- real_eval.py



## 3. IMPORTS & DEPENDENCIES PER FILE

**`lgp_eval.py`**:
- Imports: __future__.annotations, collections.defaultdict, core.gemini_llm.GeminiLLM, core.gemini_llm.get_gemini_llm, core.reflexion.ReflexionEngine, core.reflexion.get_reflexion_engine, csv, dataclasses.asdict, dataclasses.dataclass, dataclasses.field, evaluation.liar_agent.LiarAgent, evaluation.liar_agent.get_liar_agent, json, logging, math, os, sys, time, traceback, typing.Any, typing.Dict, typing.List, typing.Optional
**`setup_dataset.py`**:
- Imports: datasets.load_dataset, json
**`append_report.py`**:
- Imports: os
**`run_factored_eval.py`**:
- Imports: __future__.annotations, core.gemini_llm.GeminiLLM, core.gemini_llm.get_gemini_llm, core.reflexion.ReflexionEngine, core.reflexion.get_reflexion_engine, json, logging, math, os, sys, time, traceback, typing.Any, typing.Dict, typing.List, typing.Optional
**`run_final_eval.py`**:
- Imports: __future__.annotations, core.gemini_llm.GeminiLLM, core.gemini_llm.get_gemini_llm, core.reflexion.ReflexionEngine, core.reflexion.detect_drift_from_facts, core.reflexion.get_reflexion_engine, core.reflexion.split_into_steps, csv, json, logging, math, os, sys, time, traceback, typing.Any, typing.Dict, typing.List, typing.Optional
**`validate_pot_upgrade.py`**:
- Imports: __future__.annotations, core.pot_converter.multi_pass_validate, core.pot_converter.reasoning_to_program, core.reflexion.detect_drift_from_facts, core.reflexion.normalize_var, json, math, os, symbolic.decomposer.AtomicFact, symbolic.decomposer.SymbolicDecomposer, sys, traceback, typing.Any, typing.Dict, typing.List
**`patch_reflexion.py`**:
- Imports: re
**`main.py`**:
- Imports: __future__.annotations, argparse, core.policy.get_policy_manager, json, logging, sys, typing.List, typing.Optional
**`research_eval.py`**:
- Imports: collections.defaultdict, csv, dataclasses.asdict, dataclasses.dataclass, json, math, matplotlib, matplotlib.pyplot, os, re, shutil, symbolic.decomposer.SymbolicDecomposer, symbolic.ssce_algorithm.SSCEEnforcementError, symbolic.ssce_algorithm.get_ssce_engine, symbolic.table.get_symbolic_table, sys, time, typing.Any, typing.Dict, typing.List, typing.Optional, typing.Tuple, verifier.numeric_nli.get_numeric_consistency_gate, verifier.pot_engine.get_pot_engine
**`final_benchmark_eval.py`**:
- Imports: collections.defaultdict, core.reflexion.detect_drift_from_facts, json, math, os, re, research_eval.QueryResult, research_eval.compute_metrics, research_eval.extract_final_number, research_eval.is_correct, research_eval.normalize_gsm_query, research_eval.normalize_query, research_eval.run_baseline, research_eval.run_lgp, research_eval.safe_execute, symbolic.decomposer.SymbolicDecomposer, sys, time, typing.Any, typing.Dict, typing.List, typing.Optional, typing.Tuple
**`real_eval.py`**:
- Imports: argparse, core.gemini_llm.get_gemini_llm, core.reflexion._safe_execute, core.reflexion.partial_repair, datasets.load_dataset, dotenv.load_dotenv, evaluation.baselines.run_vanilla_baseline, evaluation.datasets.EvalSample, evaluation.runner.EvalResult, evaluation.runner.run_lgp_pipeline, json, logging, math, os, symbolic.decomposer.extract_equations, time, typing.Any, typing.Dict, typing.List, verifier.pot_engine.get_pot_engine
**`generate_fresh_results.py`**:
- Imports: csv, dataclasses.asdict, json, lgp_eval.EvalResult, lgp_eval.GSM_QUERIES, lgp_eval.RESULTS_DIR, lgp_eval.SYNTHETIC_DRIFT_QUERIES, lgp_eval.get_gemini_llm, lgp_eval.get_reflexion_engine, lgp_eval.is_correct, lgp_eval.run_baseline, lgp_eval.run_lgp, os, sys, time
**`generate_report.py`**:
- Imports: os
**`generate_full_report.py`**:
- Imports: os, sys
**`temp_real_eval.py`**:
- Imports: argparse, datasets.load_dataset, dotenv.load_dotenv, evaluation.baselines.run_vanilla_baseline, evaluation.datasets.EvalSample, evaluation.runner.run_lgp_pipeline, json, logging, os, time, typing.Any, typing.Dict, typing.List
**`reproduce_gsm_03.py`**:
- Imports: core.gemini_llm.get_gemini_llm, core.reflexion.get_reflexion_engine, logging, os, sys
**`generate_snapshot.py`**:
- Imports: ast, json, os, re, subprocess
**`gsm_dataset.py`**:
- Imports: datasets.load_dataset, json
**`scratch/inspect_svamp.py`**:
- Imports: datasets.load_dataset
**`scratch/generate_svamp.py`**:
- Imports: datasets.load_dataset, json
**`verifier/nli_gate.py`**:
- Imports: __future__.annotations, dataclasses.dataclass, logging, torch, transformers.AutoModelForSequenceClassification, transformers.AutoTokenizer, typing.List, typing.Optional, typing.Tuple
**`verifier/factored_verifier.py`**:
- Imports: __future__.annotations, dataclasses.dataclass, dataclasses.field, logging, re, symbolic.decomposer.AtomicFact, typing.Any, typing.Dict, typing.List, typing.Optional, typing.Tuple
**`verifier/numeric_nli.py`**:
- Imports: __future__.annotations, dataclasses.dataclass, logging, math, re, symbolic.table.get_symbolic_table, typing.Any, typing.Dict, typing.List, typing.Optional, typing.Tuple
**`verifier/pot_engine.py`**:
- Imports: __future__.annotations, dataclasses.dataclass, typing.Dict, typing.List
**`verifier/sandbox.py`**:
- Imports: __future__.annotations, ast, dataclasses.dataclass, docker, docker.errors.DockerException, json, logging, os, tempfile, typing.Any, typing.Dict, typing.Optional, typing.Set
**`symbolic/table.py`**:
- Imports: __future__.annotations, dataclasses.dataclass, datetime.datetime, logging, threading, typing.Any, typing.Dict, typing.Optional, typing.Tuple
**`symbolic/decomposer.py`**:
- Imports: __future__.annotations
**`symbolic/ssce_algorithm.py`**:
- Imports: __future__.annotations, dataclasses.dataclass, logging, symbolic.table.get_symbolic_table, typing.Any, typing.Dict, typing.List
**`data/popqa_loader.py`**:
- Imports: __future__.annotations, dataclasses.dataclass, json, logging, numpy, os, sentence_transformers.SentenceTransformer, typing.Any, typing.Dict, typing.List, typing.Optional
**`data/logger.py`**:
- Imports: __future__.annotations, collections.defaultdict, datetime.datetime, json, logging, symbolic.ssce_algorithm.DriftReport, symbolic.table.get_symbolic_table, typing.Any, typing.Dict, typing.List, typing.Optional
**`evaluation/runner.py`**:
- Imports: __future__.annotations, argparse, core.gemini_llm.get_gemini_llm, core.reflexion.get_reflexion_engine, dataclasses.asdict, dataclasses.dataclass, dataclasses.field, evaluation.baselines.get_baseline_runner, evaluation.datasets.EvalSample, evaluation.datasets.load_dataset_by_name, evaluation.metrics.compute_metrics, json, logging, os, time, tqdm.tqdm, typing.Any, typing.Callable, typing.Dict, typing.List, typing.Optional
**`evaluation/baselines.py`**:
- Imports: __future__.annotations, core.gemini_llm.get_gemini_llm, evaluation.datasets.EvalSample, evaluation.runner.EvalResult, json, logging, os, re, time, typing.Any, typing.Callable, typing.Dict, typing.Optional
**`evaluation/metrics.py`**:
- Imports: __future__.annotations, collections.defaultdict, evaluation.runner.EvalResult, logging, re, typing.Any, typing.Dict, typing.List, typing.Optional
**`evaluation/export.py`**:
- Imports: __future__.annotations, csv, evaluation.metrics.compute_comparative_metrics, evaluation.metrics.compute_metrics, evaluation.runner.EvalResult, json, logging, os, typing.Any, typing.Dict, typing.List, typing.Optional
**`evaluation/stress_tests.py`**:
- Imports: __future__.annotations, dataclasses.dataclass, logging, symbolic.decomposer.get_symbolic_decomposer, symbolic.ssce_algorithm.SSCEEnforcementError, symbolic.ssce_algorithm.get_ssce_engine, symbolic.table.get_symbolic_table, sys, time, typing.List, typing.Optional, verifier.pot_engine.get_pot_engine, verifier.sandbox.get_sandbox_executor
**`evaluation/liar_agent.py`**:
- Imports: __future__.annotations, dataclasses.dataclass, logging, random, re, typing.Any, typing.Dict, typing.List, typing.Optional, typing.Tuple
**`evaluation/datasets.py`**:
- Imports: __future__.annotations, dataclasses.asdict, dataclasses.dataclass, dataclasses.field, datasets.load_dataset, json, logging, re, typing.Any, typing.Dict, typing.List, typing.Optional
**`core/gemini_llm.py`**:
- Imports: core.nvidia_llm.DecompositionResult, core.nvidia_llm.NvidiaLLM, core.nvidia_llm.ReasoningResult, core.nvidia_llm.SUPPORTED_PREDICATES, core.nvidia_llm.get_nvidia_llm
**`core/pot_converter.py`**:
- Imports: __future__.annotations, logging, re, symbolic.decomposer.SymbolicDecomposer, typing.Dict, typing.List, typing.Optional, typing.Tuple
**`core/reflexion.py`**:
- Imports: __future__.annotations, ast, core.gemini_llm.DecompositionResult, core.gemini_llm.GeminiLLM, core.gemini_llm.ReasoningResult, core.nvidia_llm._safe_extract_text, dataclasses.dataclass, dataclasses.field, logging, math, re, symbolic.decomposer.AtomicFact, symbolic.decomposer.extract_equations, symbolic.decomposer.get_symbolic_decomposer, symbolic.ssce_algorithm.DriftReport, symbolic.ssce_algorithm.SSCEEnforcementError, symbolic.ssce_algorithm.SSCEEngine, symbolic.ssce_algorithm.get_ssce_engine, symbolic.table.get_symbolic_table, typing.Any, typing.Dict, typing.List, typing.Optional, typing.Tuple, verifier.factored_verifier.values_match, verifier.pot_engine.PoTScript, verifier.pot_engine.get_pot_engine
**`core/state_manager.py`**:
- Imports: __future__.annotations, dataclasses.asdict, dataclasses.dataclass, dataclasses.field, logging, symbolic.table.get_symbolic_table, typing.Any, typing.Dict, typing.List, typing.Optional
**`core/policy.py`**:
- Imports: __future__.annotations, core.state_manager.StateManager, data.logger.get_semantic_logger, data.popqa_loader.PopQALoader, logging, os, symbolic.decomposer.DecompositionComplexityError, symbolic.decomposer.get_symbolic_decomposer, symbolic.ssce_algorithm.SSCEEnforcementError, symbolic.ssce_algorithm.get_ssce_engine, symbolic.table.get_symbolic_table, time, typing.Any, typing.Dict, typing.Optional, verifier.nli_gate.get_nli_gate, verifier.numeric_nli.get_numeric_consistency_gate, verifier.pot_engine.get_pot_engine, verifier.sandbox.get_sandbox_executor
**`core/nvidia_llm.py`**:
- Imports: __future__.annotations, dataclasses.dataclass, dotenv.load_dotenv, json, logging, openai.OpenAI, os, re, threading, time, typing.Any, typing.Dict, typing.List, typing.Optional, typing.Tuple


## 4. CONFIGURATION & BUILD FILES

### `.env.example`
```text
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-pro-exp-03-25
REQUEST_DELAY=12

```

### `Dockerfile`
```text
FROM python:3.10-slim

# Install required Python libraries
RUN pip install --no-cache-dir sympy

WORKDIR /app
```

### `.gitignore`
```text
.env
__pycache__/
*.pyc
.ipynb_checkpoints/
.pytest_cache/
venv/
env/
.DS_Store
codebase_bundle.txt

```

### `requirements.txt`
```text
# HalluciNOT (LGP) — Core Dependencies
# Python 3.12.3

# ML / NLP
torch>=2.8.0
transformers>=4.57.0
sentence-transformers>=5.2.0
datasets>=4.5.0
accelerate>=1.11.0

# LLM APIs
google-generativeai>=0.8.0

# Symbolic / Math
sympy>=1.14.0
numpy>=2.2.0
scipy>=1.16.0
scikit-learn>=1.7.0

# Docker Sandbox
docker>=7.1.0

# Utilities
python-dotenv>=1.0.0
tqdm>=4.66.0
tabulate>=0.9.0

```

### `.env`
```text
# HalluciNOT (LGP) — Environment Variables

# NVIDIA API (primary LLM backend)
NVIDIA_API_KEY=REMOVED_API_KEY

# LLM Model
LLM_MODEL=openai/gpt-oss-120b

# NLI Agreement threshold
NLI_THRESHOLD=0.75

# Max Reflexion retry loops
MAX_REFLEXION_TRIALS=3

# Docker sandbox image
SANDBOX_DOCKER_IMAGE=lgp-sandbox:latest

```

## 5. STATIC & ASSET FILES

- `data/controlled_drift_dataset.json`
- `data/drift_stress_test.json`
- `data/gsm8k_subset.json`
- `data/gsmhard_subset.json`
- `data/mawps_subset.json`
- `data/svamp_subset.json`
- `gsm_subset.json`
- `project_report.md`
- `results/accuracy_comparison.png`
- `results/analysis_report.txt`
- `results/correction_success_rate.png`
- `results/debug_trace_gsm_03.txt`
- `results/demo_trace.txt`
- `results/drift_detection_rate.png`
- `results/ensemble_ablation_table.txt`
- `results/evaluation_results.csv`
- `results/factored_eval_metrics.json`
- `results/final_benchmark_metrics.json`
- `results/final_comparison_table.txt`
- `results/final_demo_trace.txt`
- `results/final_metrics.json`
- `results/final_metrics_table.txt`
- `results/final_presentation_tables.txt`
- `results/final_real_eval_table.txt`
- `results/final_report_tables.md`
- `results/improvement_vs_regression.png`
- `results/lgp_evaluation_report.md`
- `results/lgp_evaluation_results.csv`
- `results/lgp_summary_metrics.json`
- `results/mawps_real_eval_table.txt`
- `results/mawps_summary_metrics.json`
- `results/plots/accuracy_comparison.png`
- `results/plots/before_vs_after.png`
- `results/plots/drift_detection.png`
- `results/plots/drift_type_pie.png`
- `results/plots/error_breakdown.png`
- `results/plots/improvement_vs_regression.png`
- `results/plots/logic_type_breakdown.png`
- `results/pot_validation_results.json`
- `results/progress_ensemble.json`
- `results/progress_eval.json`
- `results/progress_mawps.json`
- `results/progress_svamp.json`
- `results/research_summary.txt`
- `results/results_after.csv`
- `results/results_before.csv`
- `results/summary_metrics.json`
- `results/summary_metrics_before.json`
- `results/summary_report.txt`
- `results/svamp_real_eval_table.txt`
- `results/svamp_summary_metrics.json`
- `results/temp_progress.json`
- `summary_metrics.json`
- `synthetic_drift.json`


## 6. TEST FILES

- `__pycache__/test_norm2.cpython-312.pyc`


## 7. README & DOCS

### `README.md`
```markdown
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

3. **`verifier/` (Execution 
...[truncated]
```

## 8. GIT INFO

### Last Modified Date per tracked file (git log)
```text
2026-04-17 .env.example
2026-04-08 .gitignore
2026-03-21 Dockerfile
2026-03-21 README.md
2026-03-21 core/__init__.py
2026-04-17 core/gemini_llm.py
2026-04-17 core/nvidia_llm.py
2026-04-08 core/policy.py
2026-04-17 core/pot_converter.py
2026-04-17 core/reflexion.py
2026-03-21 core/state_manager.py
2026-03-21 data/__init__.py
2026-04-08 data/controlled_drift_dataset.json
2026-04-08 data/drift_stress_test.json
2026-04-08 data/gsm8k_subset.json
2026-04-08 data/gsmhard_subset.json
2026-03-21 data/logger.py
2026-03-21 data/popqa_loader.py
2026-03-21 evaluation/__init__.py
2026-04-17 evaluation/baselines.py
2026-03-21 evaluation/datasets.py
2026-03-21 evaluation/export.py
2026-04-08 evaluation/liar_agent.py
2026-03-21 evaluation/metrics.py
2026-04-17 evaluation/runner.py
2026-03-21 evaluation/stress_tests.py
2026-04-08 final_benchmark_eval.py
2026-04-17 generate_fresh_results.py
2026-03-21 gsm_dataset.py
2026-03-21 gsm_subset.json
2026-04-17 lgp_eval.py
2026-03-21 main.py
2026-04-08 patch_reflexion.py
2026-04-17 real_eval.py
2026-04-17 reproduce_gsm_03.py
2026-03-21 requirements.txt
2026-04-08 research_eval.py
2026-04-08 results/accuracy_comparison.png
2026-04-08 results/analysis_report.txt
2026-04-08 results/correction_success_rate.png
2026-04-08 results/debug_trace_gsm_03.txt
2026-04-08 results/demo_trace.txt
2026-04-08 results/drift_detection_rate.png
2026-04-08 results/evaluation_results.csv
2026-04-08 results/factored_eval_metrics.json
2026-04-08 results/final_benchmark_metrics.json
2026-04-08 results/final_comparison_table.txt
2026-04-08 results/final_demo_trace.txt
2026-04-08 results/final_metrics.json
2026-04-08 results/final_metrics_table.txt
2026-04-08 results/final_presentation_tables.txt
2026-04-17 results/final_real_eval_table.txt
2026-04-08 results/final_report_tables.md
2026-04-08 results/improvement_vs_regression.png
2026-04-08 results/lgp_evaluation_report.md
2026-04-08 results/lgp_evaluation_results.csv
2026-04-08 results/lgp_summary_metrics.json
2026-04-08 results/plots/accuracy_comparison.png
2026-04-08 results/plots/before_vs_after.png
2026-04-08 results/plots/drift_detection.png
2026-04-08 results/plots/drift_type_pie.png
2026-04-08 results/plots/error_breakdown.png
2026-04-08 results/plots/improvement_vs_regression.png
2026-04-08 results/plots/logic_type_breakdown.png
2026-04-08 results/pot_validation_results.json
2026-04-17 results/progress_eval.json
2026-04-08 results/research_summary.txt
2026-04-08 results/results_after.csv
2026-04-08 results/results_before.csv
2026-04-08 results/summary_metrics.json
2026-04-08 results/summary_metrics_before.json
2026-04-08 results/summary_report.txt
2026-04-17 run_factored_eval.py
2026-04-17 run_final_eval.py
2026-04-17 summary_metrics.json
2026-03-21 symbolic/__init__.py
2026-04-17 symbolic/decomposer.py
2026-03-21 symbolic/ssce_algorithm.py
2026-03-21 symbolic/table.py
2026-04-08 synthetic_drift.json
2026-04-17 test_bug.py
2026-04-17 test_engine.py
2026-04-17 test_gemini.py
2026-04-17 test_norm.py
2026-04-17 test_norm2.py
2026-04-08 test_pot.py
2026-04-17 test_val.py
2026-04-17 test_validation.py
2026-04-08 validate_pot_upgrade.py
2026-03-21 verifier/__init__.py
2026-04-17 verifier/factored_verifier.py
2026-04-08 verifier/nli_gate.py
2026-03-21 verifier/numeric_nli.py
2026-04-17 verifier/pot_engine.py
2026-03-21 verifier/sandbox.py
```

### Untracked / Modified files (git status)
```text
M core/nvidia_llm.py
 M core/reflexion.py
 M data/gsmhard_subset.json
 M evaluation/runner.py
 M lgp_eval.py
 M real_eval.py
 M results/final_benchmark_metrics.json
 M results/final_comparison_table.txt
 M results/final_demo_trace.txt
 M results/final_presentation_tables.txt
 M results/final_real_eval_table.txt
 M results/progress_eval.json
 M run_final_eval.py
 M summary_metrics.json
 D test_bug.py
 D test_engine.py
 D test_gemini.py
 D test_norm.py
 D test_norm2.py
 D test_pot.py
 D test_val.py
 D test_validation.py
?? "Bibliography - Actual used/"
?? append_report.py
?? data/mawps_subset.json
?? data/svamp_subset.json
?? generate_full_report.py
?? generate_report.py
?? generate_snapshot.py
?? project_report.md
?? results/ensemble_ablation_table.txt
?? results/mawps_real_eval_table.txt
?? results/mawps_summary_metrics.json
?? results/progress_ensemble.json
?? results/progress_mawps.json
?? results/progress_svamp.json
?? results/svamp_real_eval_table.txt
?? results/svamp_summary_metrics.json
?? results/temp_progress.json
?? scratch/
?? setup_dataset.py
?? temp_real_eval.py
```

### Files not touched in 6+ months
```text

```
