#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Complete Function Pipeline - Generate requirements, implementations, and test

This script processes functions through the entire pipeline:
1. Generate natural language requirements
2. Generate no-context implementation
3. Generate full-context implementation
4. Test both implementations against the original code
"""

import json
import subprocess
import os
import sys
import logging
import argparse
import time
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_functions(file_path):
    """Load all functions from the annotated functions file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error("Error loading functions: %s", e)
        return None

def generate_requirements(function_data):
    """Generate natural language requirements for the function"""
    logger.info("Generating requirements for function: %s", function_data['name'])
    
    # Ensure output directory exists
    ensure_dir("concurrent_analysis_output/requirements")
    output_file = "concurrent_analysis_output/requirements/%s_requirements.txt" % function_data['name']
    
    # Write function data to a temporary file
    temp_file = "concurrent_analysis_output/temp_function_%s.json" % function_data['name']
    with open(temp_file, 'w') as f:
        json.dump(function_data, f)
    
    # Call the requirements generation script
    cmd = [
        'python', 
        'src/clean_llm_annotation_pipeline.py',
        '--input_file', temp_file,
        '--output_file', output_file
    ]
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
        if process.returncode != 0:
            logger.error("Failed to generate requirements: %s", stderr)
            return None
        
        # Read the generated requirements
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                requirements = f.read()
            return requirements
        else:
            logger.error("Requirements file not found: %s", output_file)
            return None
    except Exception as e:
        logger.error("Error generating requirements: %s", e)
        return None

def generate_no_context_implementation(function_data, requirements):
    """Generate implementation with no context"""
    logger.info("Generating no-context implementation for function: %s", function_data['name'])
    
    # Ensure output directory exists
    ensure_dir("concurrent_analysis_output/generated_functions")
    output_file = "concurrent_analysis_output/generated_functions/%s_no_context.java" % function_data['name']
    
    # Write function data and requirements to temporary files
    temp_function_file = "concurrent_analysis_output/temp_function_%s.json" % function_data['name']
    temp_req_file = "concurrent_analysis_output/temp_req_%s.txt" % function_data['name']
    
    with open(temp_function_file, 'w') as f:
        json.dump(function_data, f)
    
    with open(temp_req_file, 'w') as f:
        f.write(requirements)
    
    # Call the no-context generation script
    cmd = [
        'python', 
        'src/generate_function_no_context.py',
        '--function_file', temp_function_file,
        '--requirements_file', temp_req_file,
        '--output_file', output_file
    ]
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        # Clean up temporary files
        for file in [temp_function_file, temp_req_file]:
            if os.path.exists(file):
                os.remove(file)
                
        if process.returncode != 0:
            logger.error("Failed to generate no-context implementation: %s", stderr)
            return False
        
        return os.path.exists(output_file)
    except Exception as e:
        logger.error("Error generating no-context implementation: %s", e)
        return False

def generate_full_context_implementation(function_data, requirements):
    """Generate implementation with full context"""
    logger.info("Generating full-context implementation for function: %s", function_data['name'])
    
    # Ensure output directory exists
    ensure_dir("concurrent_analysis_output/generated_functions")
    output_file = "concurrent_analysis_output/generated_functions/%s_full_context.java" % function_data['name']
    
    # Write function data and requirements to temporary files
    temp_function_file = "concurrent_analysis_output/temp_function_%s.json" % function_data['name']
    temp_req_file = "concurrent_analysis_output/temp_req_%s.txt" % function_data['name']
    
    with open(temp_function_file, 'w') as f:
        json.dump(function_data, f)
    
    with open(temp_req_file, 'w') as f:
        f.write(requirements)
    
    # Call the full-context generation script
    cmd = [
        'python', 
        'src/generate_function_full_context.py',
        '--function_file', temp_function_file,
        '--requirements_file', temp_req_file,
        '--output_file', output_file
    ]
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        # Clean up temporary files
        for file in [temp_function_file, temp_req_file]:
            if os.path.exists(file):
                os.remove(file)
                
        if process.returncode != 0:
            logger.error("Failed to generate full-context implementation: %s", stderr)
            return False
        
        return os.path.exists(output_file)
    except Exception as e:
        logger.error("Error generating full-context implementation: %s", e)
        return False

def test_implementations(function_index, functions_file):
    """Test the implementations against the original code"""
    logger.info("Testing implementations for function index: %s", function_index)
    
    # Call the testing script
    cmd = [
        'python', 
        'src/test_without_modification_fixed3.py', 
        functions_file, 
        str(function_index)
    ]
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        return {
            'success': process.returncode == 0,
            'stdout': stdout,
            'stderr': stderr
        }
    except Exception as e:
        logger.error("Error testing implementations: %s", e)
        return {
            'success': False,
            'stdout': '',
            'stderr': str(e)
        }

def compare_implementations(function_data):
    """Compare the implementations using the compare_function_outputs.py script"""
    logger.info("Comparing implementations for function: %s", function_data['name'])
    
    # Get the function index from the functions file
    functions_file = "concurrent_analysis_output/annotated_functions.json"
    with open(functions_file, 'r') as f:
        functions = json.load(f)
    
    function_index = None
    for i, func in enumerate(functions):
        if func['name'] == function_data['name']:
            function_index = i
            break
    
    if function_index is None:
        logger.error("Function %s not found in %s", function_data['name'], functions_file)
        return None
    
    # Call the comparison script
    cmd = [
        'python', 
        'src/compare_function_outputs.py', 
        functions_file, 
        str(function_index)
    ]
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        return {
            'success': process.returncode == 0,
            'stdout': stdout,
            'stderr': stderr
        }
    except Exception as e:
        logger.error("Error comparing implementations: %s", e)
        return {
            'success': False,
            'stdout': '',
            'stderr': str(e)
        }

def process_function(function_index, functions, functions_file):
    """Process a single function through the entire pipeline"""
    function_data = functions[function_index]
    logger.info("Processing function %s/%s: %s", function_index, len(functions), function_data['name'])
    
    start_time = time.time()
    result = {
        'function_index': function_index,
        'function_name': function_data['name'],
        'class_name': function_data.get('class_name', ''),
        'repository': function_data.get('repository', ''),
        'file_path': function_data.get('file_path', ''),
        'start_time': datetime.now().isoformat(),
        'stages': {}
    }
    
    # Step 1: Generate requirements
    requirements = generate_requirements(function_data)
    result['stages']['requirements_generation'] = {
        'success': requirements is not None,
        'time': time.time() - start_time
    }
    
    if not requirements:
        result['status'] = 'failed'
        result['error'] = 'Failed to generate requirements'
        result['end_time'] = datetime.now().isoformat()
        result['total_time'] = time.time() - start_time
        return result
    
    # Step 2: Generate no-context implementation
    step_start = time.time()
    no_context_success = generate_no_context_implementation(function_data, requirements)
    result['stages']['no_context_generation'] = {
        'success': no_context_success,
        'time': time.time() - step_start
    }
    
    if not no_context_success:
        result['status'] = 'failed'
        result['error'] = 'Failed to generate no-context implementation'
        result['end_time'] = datetime.now().isoformat()
        result['total_time'] = time.time() - start_time
        return result
    
    # Step 3: Generate full-context implementation
    step_start = time.time()
    full_context_success = generate_full_context_implementation(function_data, requirements)
    result['stages']['full_context_generation'] = {
        'success': full_context_success,
        'time': time.time() - step_start
    }
    
    if not full_context_success:
        result['status'] = 'failed'
        result['error'] = 'Failed to generate full-context implementation'
        result['end_time'] = datetime.now().isoformat()
        result['total_time'] = time.time() - start_time
        return result
    
    # Step 4: Test implementations
    step_start = time.time()
    test_result = test_implementations(function_index, functions_file)
    result['stages']['testing'] = {
        'success': test_result['success'],
        'time': time.time() - step_start,
        'stdout': test_result['stdout'],
        'stderr': test_result['stderr']
    }
    
    # Step 5: Compare implementations
    step_start = time.time()
    compare_result = compare_implementations(function_data)
    result['stages']['comparison'] = {
        'success': compare_result['success'] if compare_result else False,
        'time': time.time() - step_start,
        'stdout': compare_result['stdout'] if compare_result else '',
        'stderr': compare_result['stderr'] if compare_result else ''
    }
    
    result['status'] = 'success'
    result['end_time'] = datetime.now().isoformat()
    result['total_time'] = time.time() - start_time
    return result

def main():
    parser = argparse.ArgumentParser(description='Process functions through the complete pipeline')
    parser.add_argument('--functions_file', default='concurrent_analysis_output/annotated_functions.json',
                        help='Path to the annotated functions JSON file')
    parser.add_argument('--start_index', type=int, default=0,
                        help='Starting function index (default: 0)')
    parser.add_argument('--end_index', type=int, default=None,
                        help='Ending function index (default: process all functions)')
    parser.add_argument('--output_file', default='concurrent_analysis_output/pipeline_results.json',
                        help='Path to save the results')
    
    args = parser.parse_args()
    
    # Load functions
    functions = load_functions(args.functions_file)
    if not functions:
        logger.error("Failed to load functions from %s", args.functions_file)
        return
    
    # Set end index
    end_index = args.end_index if args.end_index is not None else len(functions)
    
    # Load existing results if available
    results = []
    if os.path.exists(args.output_file):
        try:
            with open(args.output_file, 'r') as f:
                results = json.load(f)
            logger.info("Loaded %s existing results from %s", len(results), args.output_file)
        except Exception as e:
            logger.warning("Failed to load existing results: %s", e)
    
    # Process functions
    processed_indices = {r['function_index'] for r in results}
    for i in range(args.start_index, min(end_index, len(functions))):
        if i in processed_indices:
            logger.info("Skipping already processed function %s: %s", i, functions[i]['name'])
            continue
        
        logger.info("Processing function %s/%s: %s", i, len(functions), functions[i]['name'])
        result = process_function(i, functions, args.functions_file)
        results.append(result)
        
        # Save results after each function in case of interruption
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Print summary
    success_count = sum(1 for r in results if r.get('status') == 'success')
    logger.info("\nPipeline complete: %s/%s functions processed successfully", success_count, len(results))

if __name__ == "__main__":
    main()
