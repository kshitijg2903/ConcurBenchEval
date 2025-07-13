#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test generated function implementations against test cases
"""

import json
import os
import logging
import sys
import subprocess
import tempfile
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_function(file_path, function_index=0):
    """Load a specific function from the annotated functions file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list) and len(data) > function_index:
            return data[function_index]
        else:
            logger.error("Invalid data structure or index out of range")
            return None
    except Exception as e:
        logger.error("Error loading function: {0}".format(e))
        return None

def load_source_file(file_path):
    """Load the source file content"""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        logger.error("Error loading source file: {0}".format(e))
        return None

def load_generated_function(file_path):
    """Load the generated function content"""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        logger.error("Error loading generated function: {0}".format(e))
        return None

def replace_function_in_source(source_code, function_signature, original_function, generated_function):
    """Replace the function in the source code"""
    # Create the full function with signature and body
    original_function_full = function_signature + " {\n        " + original_function + "\n    }"
    generated_function_full = function_signature + " {\n        " + generated_function + "\n    }"
    
    # Replace the function in the source code
    return source_code.replace(original_function_full, generated_function_full)

def create_temp_directory():
    """Create a temporary directory for testing"""
    return tempfile.mkdtemp()

def copy_repository(repo_path, temp_dir):
    """Copy the repository to the temporary directory"""
    dest_path = os.path.join(temp_dir, os.path.basename(repo_path))
    shutil.copytree(repo_path, dest_path)
    return dest_path

def run_tests(repo_path, test_file_path):
    """Run the tests for the function"""
    try:
        # Change to the repository directory
        os.chdir(repo_path)
        
        # Run the tests
        result = subprocess.run(['mvn', 'test', '-Dtest=' + os.path.basename(test_file_path)], 
                              capture_output=True, text=True)
        
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        logger.error("Error running tests: {0}".format(e))
        return False, "", str(e)

def main():
    if len(sys.argv) < 4:
        logger.error("Usage: python test_generated_function.py <function_file> <function_index> <implementation_type>")
        return
    
    function_file = sys.argv[1]
    function_index = int(sys.argv[2])
    implementation_type = sys.argv[3]  # 'no_context' or 'full_context'
    
    function_data = load_function(function_file, function_index)
    if not function_data:
        logger.error("Failed to load function data")
        return
    
    # Get the function details
    function_name = function_data.get('name', '')
    signature = function_data.get('signature', '')
    file_path = function_data.get('file_path', '')
    test_files = function_data.get('test_files', [])
    
    if not test_files:
        logger.error("No test files found for function {0}".format(function_name))
        return
    
    # Load the source file
    source_code = load_source_file(file_path)
    if not source_code:
        logger.error("Failed to load source file")
        return
    
    # Load the generated function
    generated_function_path = "concurrent_analysis_output/generated_functions/{0}_{1}.java".format(
        function_name, implementation_type)
    generated_function = load_generated_function(generated_function_path)
    if not generated_function:
        logger.error("Failed to load generated function")
        return
    
    # Extract the original function from the source code
    original_function = function_data.get('source_code', '')
    if not original_function:
        logger.error("Original function source code not found")
        return
    
    # Clean up the original function (remove signature and braces)
    original_function = original_function.strip()
    if original_function.startswith(signature):
        original_function = original_function[len(signature):].strip()
    if original_function.startswith("{"):
        original_function = original_function[1:].strip()
    if original_function.endswith("}"):
        original_function = original_function[:-1].strip()
    
    # Replace the function in the source code
    modified_source = replace_function_in_source(source_code, signature, original_function, generated_function)
    
    # Create a temporary directory for testing
    temp_dir = create_temp_directory()
    logger.info("Created temporary directory: {0}".format(temp_dir))
    
    try:
        # Get the repository path
        repo_path = os.path.dirname(os.path.dirname(file_path))
        
        # Copy the repository to the temporary directory
        temp_repo_path = copy_repository(repo_path, temp_dir)
        logger.info("Copied repository to: {0}".format(temp_repo_path))
        
        # Replace the function in the copied repository
        temp_file_path = os.path.join(temp_repo_path, os.path.relpath(file_path, repo_path))
        with open(temp_file_path, 'w') as f:
            f.write(modified_source)
        logger.info("Modified source file: {0}".format(temp_file_path))
        
        # Run the tests
        for test_file in test_files:
            temp_test_file = os.path.join(temp_repo_path, os.path.relpath(test_file, repo_path))
            logger.info("Running tests from: {0}".format(temp_test_file))
            
            success, stdout, stderr = run_tests(temp_repo_path, temp_test_file)
            
            logger.info("Test result: {0}".format("PASS" if success else "FAIL"))
            logger.info("Test output:\n{0}".format(stdout))
            if stderr:
                logger.info("Test errors:\n{0}".format(stderr))
    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)
        logger.info("Cleaned up temporary directory")

if __name__ == "__main__":
    main()
