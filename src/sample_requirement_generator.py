#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sample Requirement Generator
Generates a clean requirement for a single function without using its source code
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

def generate_clean_requirement(function_data):
    """Generate a clean requirement without source code or thinking process"""
    
    # Extract only the necessary metadata
    name = function_data.get('name', 'N/A')
    signature = function_data.get('signature', 'N/A')
    file_path = function_data.get('file_path', 'N/A')
    dependencies = function_data.get('dependencies', [])
    concurrency_patterns = function_data.get('concurrency_patterns', [])
    sync_primitives = function_data.get('sync_primitives', [])
    domain = function_data.get('domain', 'N/A')
    complexity_score = function_data.get('complexity_score', 0)
    class_name = function_data.get('class_name', 'N/A')
    
    # Create a clean prompt without source code
    prompt = """Generate concise natural-language requirements for this Java function:

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
    
    print("\n=== PROMPT ===\n")
    print(prompt)
    print("\n=== END PROMPT ===\n")
    
    # For now, we'll manually create a sample clean requirement
    # In the real implementation, this would call the LLM API
    
    # Extract function parameters from signature
    params = []
    if "(" in signature and ")" in signature:
        param_section = signature.split("(")[1].split(")")[0]
        if param_section.strip():
            param_list = param_section.split(",")
            for param in param_list:
                param = param.strip()
                if param:
                    parts = param.split()
                    if len(parts) >= 2:
                        param_type = parts[0]
                        param_name = parts[-1].replace(',', '')
                        params.append((param_name, param_type))
    
    # Determine return type
    return_type = "void"
    if signature.split("(")[0].strip():
        signature_parts = signature.split("(")[0].strip().split()
        for part in signature_parts:
            if part not in ['public', 'private', 'protected', 'static', 'final', 'synchronized']:
                return_type = part
                break
    
    # Create a sample clean requirement
    clean_requirement = """Function Requirements:

1. Purpose: The {0} function manages concurrent operations in the {1} class, providing thread-safe access to shared resources.

2. Input-Output:
""".format(name, class_name)
    
    # Add parameters
    if params:
        for param_name, param_type in params:
            clean_requirement += "   - :param {0}: {1}, Parameter for {2} operation\n".format(
                param_name, param_type, name
            )
    else:
        clean_requirement += "   - No parameters\n"
    
    # Add return type
    if return_type != "void":
        clean_requirement += "   - :return: {0}, Result of the {1} operation\n".format(
            return_type, name
        )
    else:
        clean_requirement += "   - No return value\n"
    
    # Add dependencies
    clean_requirement += "\n3. Dependencies:\n"
    if dependencies:
        clean_requirement += "   - Relies on: {0}".format(", ".join(dependencies[:3]))
        if len(dependencies) > 3:
            clean_requirement += " and {0} more dependencies".format(len(dependencies) - 3)
    else:
        clean_requirement += "   - No external dependencies"
    
    # Add concurrency patterns
    clean_requirement += "\n\n4. Concurrency patterns:\n"
    if concurrency_patterns or sync_primitives:
        if concurrency_patterns:
            clean_requirement += "   - Uses {0} patterns for thread safety\n".format(
                ", ".join(concurrency_patterns)
            )
        if sync_primitives:
            clean_requirement += "   - Employs {0} synchronization primitives".format(
                ", ".join(sync_primitives)
            )
    else:
        clean_requirement += "   - Basic thread safety mechanisms"
    
    return clean_requirement

def main():
    if len(sys.argv) < 2:
        logger.error("Please provide a file path to an annotated function JSON file")
        return
    
    file_path = sys.argv[1]
    function_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    function_data = load_function(file_path, function_index)
    if function_data:
        clean_requirement = generate_clean_requirement(function_data)
        print("\n=== CLEAN REQUIREMENT ===\n")
        print(clean_requirement)
        print("\n=== END CLEAN REQUIREMENT ===\n")
    else:
        logger.error("Failed to load function data")

if __name__ == "__main__":
    main()
