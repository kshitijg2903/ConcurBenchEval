#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test a single function annotation
"""

import json
import os
import logging
import sys
import requests
import time

# Constants
API_URL = "https://api.together.xyz/v1/chat/completions"
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"

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

def create_prompt(fn):
    """Create a clean prompt without source code"""
    name = fn.get('name', 'N/A')
    signature = fn.get('signature', 'N/A')
    file_path = fn.get('file_path', 'N/A')
    dependencies = fn.get('dependencies', [])
    concurrency_patterns = fn.get('concurrency_patterns', [])
    sync_primitives = fn.get('sync_primitives', [])
    domain = fn.get('domain', 'N/A')
    complexity_score = fn.get('complexity_score', 0)
    class_name = fn.get('class_name', 'N/A')
    
    return """Generate concise natural-language requirements for this Java function:

**Function Metadata**
- `name`: {0}
- `signature`: {1}
- `file_path`: {2}
- `class_name`: {3}
- `dependencies`: {4}
- `concurrency_patterns`: {5}
- `sync_primitives`: {6}
- `domain`: {7}
- `complexity_score`: {8}

Follow this format:

Function Requirements:

1. Purpose: Brief technical summary of what the function does.

2. Input-Output:
   - :param name: type, description
   - :return: type, description

3. Dependencies: List what it relies on.

4. Concurrency patterns: Describe any mechanisms used.

Generate the requirements now:""".format(
        name, signature, file_path, class_name, dependencies, 
        concurrency_patterns, sync_primitives, domain, complexity_score
    )

def call_llm_api(prompt, api_key):
    """Call the Together AI API"""
    headers = {
        "Authorization": "Bearer {0}".format(api_key),
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000,
        "temperature": 0.3,
        "top_p": 0.9,
    }

    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
        else:
            logger.error("API error %s - %s", resp.status_code, resp.text)
    except Exception as exc:
        logger.error("Request failed: %s", exc)

    return None

def clean_response(response):
    """Clean the LLM response to remove thinking process"""
    if not response:
        return "Annotation failed"
    
    # Remove thinking process if present
    if "<think>" in response and "</think>" in response:
        parts = response.split("</think>")
        if len(parts) > 1:
            return parts[1].strip()
    
    # If no thinking process, return as is
    return response

def main():
    if len(sys.argv) < 3:
        logger.error("Usage: python test_single_function.py <file_path> <function_index> <api_key>")
        return
    
    file_path = sys.argv[1]
    function_index = int(sys.argv[2])
    api_key = sys.argv[3]
    
    function_data = load_function(file_path, function_index)
    if function_data:
        prompt = create_prompt(function_data)
        print("\n=== PROMPT ===\n")
        print(prompt)
        print("\n=== END PROMPT ===\n")
        
        raw_response = call_llm_api(prompt, api_key)
        clean_req = clean_response(raw_response)
        
        print("\n=== RAW RESPONSE ===\n")
        print(raw_response)
        print("\n=== END RAW RESPONSE ===\n")
        
        print("\n=== CLEAN REQUIREMENT ===\n")
        print(clean_req)
        print("\n=== END CLEAN REQUIREMENT ===\n")
    else:
        logger.error("Failed to load function data")

if __name__ == "__main__":
    main()
