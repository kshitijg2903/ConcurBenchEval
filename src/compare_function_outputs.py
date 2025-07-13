#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare the outputs of the original function and the generated functions
"""

import json
import os
import logging
import sys

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

def load_generated_function(file_path):
    """Load the generated function content"""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        logger.error("Error loading generated function: {0}".format(e))
        return None

def extract_function_body(source_code):
    """Extract the function body from the source code"""
    # Remove leading and trailing whitespace
    source_code = source_code.strip()
    
    # If the source code starts with a signature, remove it
    if source_code.startswith("public") or source_code.startswith("private") or source_code.startswith("protected"):
        # Find the opening brace
        brace_index = source_code.find("{")
        if brace_index != -1:
            source_code = source_code[brace_index + 1:].strip()
    
    # If the source code ends with a closing brace, remove it
    if source_code.endswith("}"):
        source_code = source_code[:-1].strip()
    
    return source_code

def compare_functions(original, no_context, full_context):
    """Compare the original function with the generated functions"""
    # Clean up the functions
    original = extract_function_body(original)
    no_context = extract_function_body(no_context)
    full_context = extract_function_body(full_context)
    
    # Compare the functions
    no_context_match = original.strip() == no_context.strip()
    full_context_match = original.strip() == full_context.strip()
    
    return no_context_match, full_context_match

def main():
    if len(sys.argv) < 2:
        logger.error("Usage: python compare_function_outputs.py <function_file> <function_index>")
        return
    
    function_file = sys.argv[1]
    function_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    function_data = load_function(function_file, function_index)
    if not function_data:
        logger.error("Failed to load function data")
        return
    
    # Get the function details
    function_name = function_data.get('name', '')
    original_source = function_data.get('source_code', '')
    
    # Load the generated functions
    no_context_path = "concurrent_analysis_output/generated_functions/{0}_no_context.java".format(function_name)
    full_context_path = "concurrent_analysis_output/generated_functions/{0}_full_context.java".format(function_name)
    
    no_context = load_generated_function(no_context_path)
    full_context = load_generated_function(full_context_path)
    
    if not no_context or not full_context:
        logger.error("Failed to load generated functions")
        return
    
    # Compare the functions
    no_context_match, full_context_match = compare_functions(original_source, no_context, full_context)
    
    print("\n=== FUNCTION COMPARISON RESULTS ===\n")
    print("Original function:")
    print(original_source.strip())
    print("\nNo-context generated function:")
    print(no_context.strip())
    print("\nFull-context generated function:")
    print(full_context.strip())
    print("\nNo-context match: {0}".format("YES" if no_context_match else "NO"))
    print("Full-context match: {0}".format("YES" if full_context_match else "NO"))
    
    # Determine which implementation is better
    if no_context_match and full_context_match:
        print("\nBoth implementations match the original function.")
    elif no_context_match:
        print("\nThe no-context implementation matches the original function.")
    elif full_context_match:
        print("\nThe full-context implementation matches the original function.")
    else:
        print("\nNeither implementation matches the original function.")
        
        # Compare the implementations to see which is closer
        no_context_words = set(no_context.strip().split())
        full_context_words = set(full_context.strip().split())
        original_words = set(original_source.strip().split())
        
        no_context_similarity = len(no_context_words.intersection(original_words)) / len(original_words)
        full_context_similarity = len(full_context_words.intersection(original_words)) / len(original_words)
        
        print("No-context similarity: {0:.2f}".format(no_context_similarity))
        print("Full-context similarity: {0:.2f}".format(full_context_similarity))
        
        if no_context_similarity > full_context_similarity:
            print("The no-context implementation is closer to the original function.")
        elif full_context_similarity > no_context_similarity:
            print("The full-context implementation is closer to the original function.")
        else:
            print("Both implementations are equally similar to the original function.")

if __name__ == "__main__":
    main()
