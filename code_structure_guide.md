# ConcurBench Code Structure Guide

This document provides a quick reference to the main files and components of the ConcurBench project, organized by pipeline stage. Use this guide to understand where to find specific functionality and how the different components work together.

## Project Overview

ConcurBench is a benchmark framework for evaluating LLM-generated concurrent code. The project follows a pipeline architecture with these main stages:

1. Repository discovery and collection
2. Repository cloning and setup
3. Function discovery and extraction
4. Test discovery and verification
5. Static analysis and filtering
6. LLM annotation pipeline
7. Function generation with different context levels
8. Evaluation framework

## Core Files by Pipeline Stage

### Phase 1-2: Repository Discovery, Collection, and Setup

- **`src/github_scraper.py`**: Discovers high-quality repositories with concurrent code
- **`src/enhanced_analyzer.py`**: Main analyzer class that handles repository cloning and initial setup
  - Key method: `clone_selected_repositories()` - Clones repositories based on quality criteria
  - Key method: `analyze_java_files()` - Initial analysis of Java files in repositories

### Phase 3-4: Function and Test Discovery

- **`src/enhanced_analyzer.py`**: 
  - Key method: `extract_java_concurrent_functions()` - Extracts concurrent functions from Java files
  - Key method: `_link_test_files()` - Links test files to concurrent functions
  - Key method: `_extract_test_functions()` - Extracts test functions from test files

### Phase 5: Static Analysis and Filtering

- **`src/enhanced_analyzer.py`**: 
  - Key method: `_calculate_complexity_score()` - Calculates complexity scores for functions
  - Key method: `_classify_function_domain()` - Classifies functions by domain
  - Key method: `save_detailed_results()` - Saves analysis results to JSON files

### Phase 6: LLM Annotation Pipeline

- **`src/llm_annotation_pipeline.py`**: Generates natural language requirements for functions
  - Key method: `create_prompt()` - Creates prompts for LLM annotation
  - Key method: `call_llm_api()` - Calls the LLM API to generate annotations
  - Key method: `annotate_all()` - Orchestrates the annotation process

- **`src/clean_llm_annotation_pipeline.py`**: Simplified version for processing single functions
  - Used for generating requirements for individual functions

### Phase 7: Function Generation with Different Context Levels

- **`src/generate_function_no_context.py`**: Generates function implementations with no context
  - Key method: `generate_implementation()` - Generates implementation using only requirements

- **`src/generate_function_full_context.py`**: Generates function implementations with full context
  - Key method: `extract_function_context()` - Extracts context from the file
  - Key method: `generate_implementation()` - Generates implementation using requirements and context

### Phase 8: Evaluation Framework

- **`src/test_single_function.py`**: Tests a single function annotation
  - Used for debugging and testing individual functions

- **`src/test_generated_function.py`**: Tests LLM-generated function implementations
  - Key method: `create_test_harness()` - Creates a test harness for the function

- **`src/compare_function_outputs.py`**: Compares original and generated function outputs
  - Key method: `compare_functions()` - Compares implementations for correctness

- **`src/test_without_modification_fixed3.py`**: Tests functions without modifying original code
  - Key method: `create_test_environment()` - Creates isolated test environment
  - Key method: `run_tests()` - Runs tests on generated implementations

## End-to-End Orchestration

- **`src/complete_function_pipeline.py`**: Main orchestration script for the entire pipeline
  - Key method: `process_function()` - Processes a single function through all stages
  - Key method: `main()` - Orchestrates the entire pipeline with checkpoint-based processing

## Output Structure

- **`concurrent_analysis_output/repositories/`**: Contains analysis results for each repository
- **`concurrent_analysis_output/requirements/`**: Contains generated requirements for functions
- **`concurrent_analysis_output/generated_functions/`**: Contains LLM-generated implementations
- **`concurrent_analysis_output/pipeline_results.json`**: Contains results of the evaluation pipeline

## Key Implementation Details

### Dynamic Test Harness Generation

The dynamic test harness generation is implemented in `test_without_modification_fixed3.py`. This file contains the logic for:
- Creating isolated testing environments
- Implementing the multi-stage compilation strategy
- Developing the class substitution technique

Key methods to review:
- `create_test_environment()`: Sets up the isolated test environment
- `create_test_harness()`: Generates the test harness code
- `run_tests()`: Executes tests with fallback mechanisms

### Orchestration Wrapper Script

The orchestration wrapper script is implemented in `complete_function_pipeline.py`. This file contains:
- Checkpoint-based processing logic
- Pipeline stage integration
- Progress tracking and resumption capabilities

Key methods to review:
- `process_function()`: Processes a single function through all pipeline stages
- `main()`: Handles command-line arguments and orchestrates the entire pipeline

## Getting Started with the Code

If you're new to the project and want to understand the implementation of the key "hacks" mentioned in our documentation:

1. Start with `complete_function_pipeline.py` to understand the overall orchestration
2. Review `test_without_modification_fixed3.py` to see the dynamic test harness generation
3. Look at `generate_function_no_context.py` and `generate_function_full_context.py` to understand the different context levels
4. Examine `llm_annotation_pipeline.py` to see how requirements are generated

These files contain the core implementation of the creative solutions described in our documentation.
