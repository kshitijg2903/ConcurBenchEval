**ConcurBench**: 

ConcurBench is a benchmark framework designed to evaluate the accuracy of Large Language Models (LLMs) in generating concurrent code. The project collects high-quality concurrent code examples from popular open-source repositories, annotates them with natural language requirements, and tests LLMs' ability to regenerate these functions with varying levels of context. This provides a standardized way to measure how well AI models can handle the complexities of concurrent programming.

**Project Overview**
Goal: To build a robust dataset and evaluation framework that tests LLMs on real-world concurrency tasks, simulating realistic developer scenarios with varying context levels.

**Pipeline Overview**
Phase 1: Repository Discovery & Collection
Select high-quality repositories (e.g., 50+ stars, active development).

Analyze repository structure to identify concurrent files.

Filter by programming language, file types, and concurrency indicators (e.g., synchronized, mutex, Thread, etc.).

**Phase 2: Repository Cloning & Setup**
Clone the selected repositories locally.

Organize directory structure for easy access.

Verify code and test file accessibility.

**Phase 3: Function Discovery & Extraction**
Parse source files to extract functions involving concurrency.

Detect synchronization patterns (e.g., locks, atomic variables).

Extract function signatures, dependencies, and compute complexity metrics.

**Phase 4: Test Discovery & Verification**
Identify and match test cases corresponding to each function.

Validate test presence, coverage, and relevance.

Tag functions based on test quality (e.g., high-coverage, weak-assertion, etc.).

**Phase 5: Static Analysis & Filtering**
Apply static code analysis to compute:

Cyclomatic complexity

Lines of code

Dependency depth

Filter out trivial, duplicate, or noisy functions.

**Phase 6: LLM Annotation Pipeline**
Generate high-quality natural language (NL) descriptions.

Provide detailed problem statements, usage hints, and expected behavior.

Annotate each function with domain tags (e.g., networking, data structures, concurrency primitives).

**Phase 7: Dataset Preparation**
Organize the final dataset with rich metadata:

Programming language

Function complexity

Domain

Function + test mapping

Structure into tiers (e.g., Easy, Medium, Hard).

**Phase 8: Evaluation Framework**
Evaluate LLM performance using varying levels of context:

Level 1: No context (function signature only)

Level 2: Local context (neighboring functions and imports)

Level 3: Full file context

Report performance using:

Pass@k

Recall@k

Functional correctness via test execution
