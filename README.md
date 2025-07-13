ConcurBench: Benchmarking Code Generation for Concurrent Functions
ConcurBench is a comprehensive benchmarking pipeline designed to evaluate the capabilities of Large Language Models (LLMs) on realistic concurrent programming tasks. It systematically extracts, annotates, and evaluates real-world concurrent functions from high-quality GitHub repositories using a modular, multi-phase pipeline.

Project Overview
Goal: To build a robust dataset and evaluation framework that tests LLMs on real-world concurrency tasks, simulating realistic developer scenarios with varying context levels.

Pipeline Overview
Phase 1: Repository Discovery & Collection 
Select high-quality repositories (e.g., 50+ stars, active development).

Analyze repo structure to identify concurrent files.

Filter by programming language, file types, and concurrency indicators (e.g., synchronized, mutex, Thread, etc.).

Phase 2: Repository Cloning & Setup
Clone the selected repositories locally.

Organize directory structure for easy access.

Verify code and test file accessibility.

Phase 3: Function Discovery & Extraction
Parse source files to extract functions involving concurrency.

Detect synchronization patterns (e.g., locks, atomic variables).

Extract function signatures, dependencies, and compute complexity metrics.

Phase 4: Test Discovery & Verification
Identify and match test cases corresponding to each function.

Validate test presence, coverage, and relevance.

Tag functions based on test quality (e.g., high-coverage, weak-assertion, etc.).

Phase 5: Static Analysis & Filtering
Apply static code analysis to compute:

Cyclomatic complexity

Lines of code

Dependency depth

Filter out trivial, duplicate, or noisy functions.

Phase 6: LLM Annotation Pipeline
Generate high-quality NL (natural language) descriptions.

Provide detailed problem statements, usage hints, and expected behavior.

Annotate each function with domain tags (e.g., networking, data structures).

Phase 7: Dataset Preparation
Organize the final dataset with rich metadata:

Language

Complexity

Domain

Function + test mapping

Structure into tiers (e.g., Easy, Medium, Hard).

Phase 8: Evaluation Framework
Evaluate LLM performance using varying levels of context:

Level 1: No context (function signature only)

Level 2: Local context (neighboring functions and imports)

Level 3: Full file context

Report performance using:

Pass@k

Recall@k

Functional correctness via test execution

