Complete EvoCodeBench Pipeline - Step by Step

Phase 1: Repository Discovery & Collection ✅ 

Gather high-quality repositories (50+ stars, active development)
Analyze repository structure for concurrent files
Filter by language, file types, and concurrency indicators

Phase 2: Repository Cloning & Setup

Clone selected repositories locally
Organize repository structure
Verify file accessibility

Phase 3: Function Discovery & Extraction

Parse source code to extract concurrent functions
Identify synchronization primitives and patterns
Extract function signatures, dependencies, and complexity

Phase 4: Test Discovery & Verification

Find corresponding test files for each function
Match functions to their test cases
Verify test coverage and quality

Phase 5: Static Analysis & Filtering

Filter functions based on quality criteria
Calculate complexity scores
Remove duplicates and low-quality functions

Phase 6: LLM Annotation Pipeline

Generate natural language requirements
Create function descriptions
Add domain labels and categorization

Phase 7: Dataset Preparation

Structure final dataset with all metadata
Organize by difficulty levels and domains
Prepare test cases and reference implementations

Phase 8: Evaluation Framework

Test LLMs with different context levels:

Level 1: No context (function signature only)
Level 2: Local context (surrounding functions/imports)
Level 3: Full context (entire file context)


Measure using Pass@k, Recall@k, and functional correctness
