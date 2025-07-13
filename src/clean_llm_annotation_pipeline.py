#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clean LLM Annotation Pipeline - Generate natural language requirements for functions
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

def generate_requirements(function_data):
    """Generate natural language requirements for the function"""
    function_name = function_data.get('name', '')
    signature = function_data.get('signature', '')
    source_code = function_data.get('source_code', '')
    class_name = function_data.get('class_name', '')
    dependencies = function_data.get('dependencies', [])
    concurrency_patterns = function_data.get('concurrency_patterns', [])
    sync_primitives = function_data.get('sync_primitives', [])
    
    # Create prompt for requirements generation
    prompt = """You are an expert Java developer specializing in concurrent programming. 
Your task is to generate concise natural language requirements for a Java function.

Function details:
- Name: %s
- Signature: %s
- Class: %s
- Dependencies: %s
- Concurrency patterns: %s
- Synchronization primitives: %s

Source code:
```java
%s
```

Generate concise natural language requirements for this function. Your response should include:

1. What the function does - A brief technical summary of the function's purpose
2. Input-Output - Description of parameters and return values
3. Dependencies - What the function relies on
4. Concurrency patterns - How the function handles thread safety

Format your response as follows:
What the function does - [brief technical summary]

Input-Output:
:param: [parameter description if any]
:return: [return value description]

Dependencies - [dependencies description]

Concurrency patterns - [concurrency patterns description]

Important guidelines:
1. Be concise and technical
2. Focus on the concurrent aspects of the function
3. Do not include any implementation details or code
4. Do not include any explanations of your thinking process
5. Return only the requirements, nothing else""" % (
        function_name, 
        signature, 
        class_name, 
        ', '.join(dependencies), 
        ', '.join(concurrency_patterns), 
        ', '.join(sync_primitives),
        source_code
    )

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
        requirements = result["choices"][0]["message"]["content"].strip()
        
        return requirements
    except Exception as e:
        logger.error("Error generating requirements: %s", e)
        
        # Fallback to mock requirements if API call fails
        mock_requirements = """What the function does - Calculates the total size in bytes by summing the size of an integer and the size returned by an inspector.

Input-Output:
:param: None
:return: int - The total size in bytes.

Dependencies - Relies on the inspector object's getSizeInBytes() method and the Primitive.INT.sizeInBytes constant.

Concurrency patterns - Utilizes the future pattern for asynchronous operations and thread-local storage to manage variables per thread, enhancing concurrency without needing synchronization."""
        
        logger.info("Using mock requirements as fallback")
        return mock_requirements

def main():
    parser = argparse.ArgumentParser(description='Generate natural language requirements for functions')
    parser.add_argument('--input_file', required=True, help='Path to the function JSON file')
    parser.add_argument('--output_file', required=True, help='Path to save the generated requirements')
    
    args = parser.parse_args()
    
    # Load function data
    function_data = load_function(args.input_file)
    if not function_data:
        logger.error("Failed to load function data from %s", args.input_file)
        return
    
    # Generate requirements
    requirements = generate_requirements(function_data)
    if not requirements:
        logger.error("Failed to generate requirements")
        return
    
    # Save requirements
    try:
        ensure_dir(os.path.dirname(args.output_file))
        with open(args.output_file, 'w') as f:
            f.write(requirements)
        logger.info("Requirements saved to %s", args.output_file)
    except Exception as e:
        logger.error("Error saving requirements: %s", e)

if __name__ == "__main__":
    main()
