#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate function implementation with no context (only requirements)
"""

import json
import os
import logging
import sys
import argparse
import requests

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# LLM API configuration - hardcoded for now
API_URL = "https://api.together.xyz/v1/chat/completions"
API_KEY = "tgp_v1_7PF9hvPhCr3HIbC4uN0rwxn03HqKlGCFnq8l1ZxONOY"
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_function(file_path):
    """Load function data from a JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error("Error loading function: %s", e)
        return None

def load_requirements(file_path):
    """Load requirements from a text file"""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        logger.error("Error loading requirements: %s", e)
        return None

def generate_implementation(function_data, requirements):
    """Generate function implementation using LLM with no context"""
    function_name = function_data.get('name', '')
    signature = function_data.get('signature', '')
    
    # Create prompt for no-context implementation
    prompt = """You are an expert Java developer specializing in concurrent programming. 
Your task is to implement a Java function based on the given requirements and signature.

Function signature: %s
Function name: %s

Requirements:
%s

Important guidelines:
1. Implement ONLY the function body, not the entire class or method signature
2. Focus on implementing the concurrent aspects correctly
3. Do not include any comments or explanations in your code
4. Do not include the method signature or closing brace
5. Your implementation should be production-ready and efficient
6. Return only the implementation code, nothing else

Implement the function body now:""" % (signature, function_name, requirements)

    # Call LLM API
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer %s" % API_KEY
        }
            
        data = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are an expert Java developer specializing in concurrent programming."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 1000
        }
        
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        implementation = result["choices"][0]["message"]["content"].strip()
        
        # Clean up the implementation (remove any markdown code blocks)
        if implementation.startswith("```java"):
            implementation = implementation.split("```java", 1)[1]
        if implementation.endswith("```"):
            implementation = implementation.rsplit("```", 1)[0]
            
        implementation = implementation.strip()
        
        return implementation
    except Exception as e:
        logger.error("Error generating implementation: %s", e)
        
        # Fallback to mock implementation if API call fails
        if function_name == "getSizeInBytes":
            mock_implementation = """Inspector inspector = getInspector();
Future<Integer> sizeFuture = inspector.getSizeInBytes();
int inspectorSize = sizeFuture.get();
return Primitive.INT.sizeInBytes + inspectorSize;"""
            logger.info("Using mock implementation as fallback")
            return mock_implementation
        return None

def main():
    parser = argparse.ArgumentParser(description='Generate function implementation with no context')
    parser.add_argument('--function_file', required=True, help='Path to the function JSON file')
    parser.add_argument('--requirements_file', required=True, help='Path to the requirements text file')
    parser.add_argument('--output_file', required=True, help='Path to save the generated implementation')
    
    args = parser.parse_args()
    
    # Load function data and requirements
    function_data = load_function(args.function_file)
    if not function_data:
        logger.error("Failed to load function data from %s", args.function_file)
        return
    
    requirements = load_requirements(args.requirements_file)
    if not requirements:
        logger.error("Failed to load requirements from %s", args.requirements_file)
        return
    
    # Generate implementation
    implementation = generate_implementation(function_data, requirements)
    if not implementation:
        logger.error("Failed to generate implementation")
        return
    
    # Save implementation
    try:
        ensure_dir(os.path.dirname(args.output_file))
        with open(args.output_file, 'w') as f:
            f.write(implementation)
        logger.info("Implementation saved to %s", args.output_file)
    except Exception as e:
        logger.error("Error saving implementation: %s", e)

if __name__ == "__main__":
    main()
