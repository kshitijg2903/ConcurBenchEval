#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mock LLM Annotation Pipeline - Phase 6 (For Testing)
Simulates generating natural-language requirements for concurrent Java functions
without requiring an actual API key. Uses templates based on function metadata.

Usage:
python mock_llm_annotation_pipeline.py --batch-size 50
"""

import argparse
import json
import logging
import os
import time
import random
from datetime import datetime

# Constants
PROMPT_VERSION = "mock-v1-20250628"
DEFAULT_MODEL_ID = "mock-llama-3-70b"
MAX_SOURCE_LINES = 400
BATCH_DIR = os.path.join("concurrent_analysis_output")
LOG_FILE = os.path.join(BATCH_DIR, "mock_llm_annotation.log")

# Logging setup
if not os.path.exists(BATCH_DIR):
    os.makedirs(BATCH_DIR)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Pipeline class
class MockLLMAnnotationPipeline:
    def __init__(self, model=DEFAULT_MODEL_ID, request_delay=0.1, max_retries=3):
        self.model = model
        self.request_delay = request_delay
        self.max_retries = max_retries

        self.analysis_files = [
            "concurrent_analysis_output/repositories/JCTools_JCTools_analysis.json",
            "concurrent_analysis_output/repositories/peptos_traffic-shm_analysis.json",
            "concurrent_analysis_output/repositories/pramalhe_ConcurrencyFreaks_analysis.json",
            "concurrent_analysis_output/repositories/RobAustin_low-latency-primitive-concurrent-queues_analysis.json",
        ]

        self.output_file = os.path.join(BATCH_DIR, "mock_annotated_functions.json")
        
        # Templates for different domains
        self.domain_templates = {
            "data_structures": [
                "This function implements a thread-safe {0} operation in a concurrent data structure.",
                "Provides a concurrent implementation of {0} for thread-safe data access.",
                "Manages concurrent access to {0} data structure with thread safety guarantees."
            ],
            "producer_consumer": [
                "Implements a {0} operation in a producer-consumer pattern.",
                "Handles the {0} operation within a concurrent producer-consumer workflow.",
                "Manages thread-safe {0} operations in a producer-consumer context."
            ],
            "atomic_operations": [
                "Performs atomic {0} operation with thread safety guarantees.",
                "Implements lock-free {0} using atomic operations.",
                "Executes {0} as an atomic operation to ensure thread safety."
            ],
            "synchronization": [
                "Provides synchronization mechanism for {0} across multiple threads.",
                "Implements thread coordination for {0} using synchronization primitives.",
                "Manages thread synchronization during {0} operations."
            ],
            "memory_management": [
                "Handles concurrent memory management for {0} operations.",
                "Manages thread-safe memory access during {0}.",
                "Implements memory consistency guarantees for concurrent {0}."
            ],
            "thread_management": [
                "Manages thread lifecycle during {0} operations.",
                "Controls thread execution for concurrent {0}.",
                "Coordinates multiple threads for efficient {0} execution."
            ],
            "utilities": [
                "Provides utility function {0} with thread safety guarantees.",
                "Implements helper method {0} for concurrent operations.",
                "Offers thread-safe utility {0} for concurrent contexts."
            ],
            "general": [
                "Implements {0} with thread safety considerations.",
                "Provides concurrent implementation of {0}.",
                "Handles {0} operation in a multi-threaded environment."
            ]
        }
        
        # Templates for concurrency patterns
        self.pattern_templates = {
            "thread_local_storage": "Uses thread-local storage to maintain thread-confined data.",
            "immutable_object": "Implements immutability to ensure thread safety without synchronization.",
            "future_pattern": "Uses future pattern for asynchronous result handling.",
            "double_checked_locking": "Implements double-checked locking for lazy initialization with reduced synchronization overhead.",
            "singleton_pattern": "Ensures thread-safe singleton instance creation and access.",
            "compare_and_swap": "Uses compare-and-swap operations for lock-free updates.",
            "producer_consumer": "Implements producer-consumer pattern for concurrent data processing.",
            "lock_free_algorithm": "Uses lock-free algorithm to avoid blocking synchronization.",
            "reader_writer_lock": "Implements reader-writer lock pattern to optimize concurrent read access.",
            "observer_pattern": "Uses thread-safe observer pattern for event notification."
        }

    # Mock annotation generator
    def generate_mock_annotation(self, fn):
        """Generate a mock annotation based on function metadata"""
        name = fn.get('name', 'function')
        domain = fn.get('domain', 'general')
        patterns = fn.get('concurrency_patterns', [])
        sync_primitives = fn.get('sync_primitives', [])
        
        # Select template based on domain
        domain_templates = self.domain_templates.get(domain, self.domain_templates['general'])
        summary = random.choice(domain_templates).format(name)
        
        # Generate input-output section
        params = []
        if "(" in fn.get('signature', ''):
            param_section = fn.get('signature', '').split('(')[1].split(')')[0]
            if param_section:
                param_list = param_section.split(',')
                for i, param in enumerate(param_list):
                    param = param.strip()
                    if param:
                        param_parts = param.split()
                        if len(param_parts) >= 2:
                            param_type = param_parts[0]
                            param_name = param_parts[-1].replace(',', '')
                            params.append(":param {0}: {1}, Parameter for {2} operation".format(
                                param_name, param_type, name))
        
        return_type = "void"
        if fn.get('signature', '').split('(')[0].strip():
            signature_parts = fn.get('signature', '').split('(')[0].strip().split()
            for part in signature_parts:
                if part not in ['public', 'private', 'protected', 'static', 'final', 'synchronized']:
                    return_type = part
                    break
        
        if return_type != "void":
            return_line = ":return: {0}, Result of the {1} operation".format(return_type, name)
        else:
            return_line = "No return value."
        
        # Generate dependencies section
        dependencies = fn.get('dependencies', [])
        if dependencies:
            deps_text = "Depends on: " + ", ".join(dependencies[:3])
            if len(dependencies) > 3:
                deps_text += ", and {0} more dependencies".format(len(dependencies) - 3)
        else:
            deps_text = "No external dependencies."
        
        # Generate concurrency patterns section
        patterns_text = []
        for pattern in patterns[:2]:
            if pattern in self.pattern_templates:
                patterns_text.append(self.pattern_templates[pattern])
        
        if sync_primitives:
            primitives_text = "Uses {0} for thread safety".format(", ".join(sync_primitives[:3]))
            if len(sync_primitives) > 3:
                primitives_text += " and {0} more synchronization primitives".format(len(sync_primitives) - 3)
            patterns_text.append(primitives_text)
        
        if not patterns_text:
            patterns_text = ["Implements basic thread safety using standard Java concurrency mechanisms."]
        
        # Combine all sections
        annotation = "{0}\n\n".format(summary)
        if params:
            annotation += "\n".join(params) + "\n"
        annotation += "{0}\n\n".format(return_line)
        annotation += "{0}\n\n".format(deps_text)
        annotation += "Concurrency patterns:\n- " + "\n- ".join(patterns_text)
        
        return annotation

    # Disk helpers
    @staticmethod
    def load_functions(path):
        if not os.path.exists(path):
            logger.warning("File not found: %s", path)
            return []
        try:
            with open(path, 'r') as fh:
                data = json.load(fh)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "functions" in data:
                return data["functions"]
            logger.error("Unexpected JSON structure in %s", path)
        except Exception as exc:
            logger.error("Failed reading %s - %s", path, exc)
        return []

    # Main driver
    def annotate_all(self, batch_size=50):
        logger.info("=== Phase 6 - Mock LLM Annotation (%s) ===", PROMPT_VERSION)

        # Build work-list
        functions = []
        for af in self.analysis_files:
            functions.extend(self.load_functions(af))
        if not functions:
            logger.error("No functions loaded - aborting.")
            return
        logger.info("Total functions to process: %d", len(functions))

        annotated = []
        now_stamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")

        for start in range(0, len(functions), batch_size):
            batch = functions[start:start + batch_size]
            batch_idx = start // batch_size + 1
            batch_tag = "{0}_batch{1}".format(now_stamp, batch_idx)
            logger.info("Batch %s (%d functions)", batch_tag, len(batch))

            batch_annotated = []
            for idx, fn in enumerate(batch, start=start + 1):
                logger.info("   - %d/%d  %s", idx, len(functions), fn.get("name"))
                content = self.generate_mock_annotation(fn)

                ann = dict(fn)  # Copy the original function data
                ann.update({
                    "llm_annotation": content,
                    "annotation_error": False,
                    "annotation_timestamp": datetime.now().isoformat(),
                    "model_used": self.model,
                    "prompt_version": PROMPT_VERSION,
                })
                batch_annotated.append(ann)
                annotated.append(ann)
                time.sleep(self.request_delay)

            # save batch
            batch_file = os.path.join(BATCH_DIR, "mock_annotated_{0}.json".format(batch_tag))
            with open(batch_file, 'w') as fh:
                json.dump(batch_annotated, fh, indent=2)
            logger.info("   Saved %s", batch_file)

        # save full corpus
        with open(self.output_file, 'w') as fh:
            json.dump(annotated, fh, indent=2)
        logger.info("All annotations written to %s", self.output_file)

        # summary
        self.generate_summary(annotated)

    # Summary stats
    def generate_summary(self, annotated):
        total = len(annotated)
        failed = sum(1 for f in annotated if f.get("annotation_error"))
        success = total - failed
        repo_stats = {}
        domain_stats = {}

        for fn in annotated:
            repo = fn.get("repository", "unknown")
            domain = fn.get("domain", "unknown")
            repo_stats[repo] = repo_stats.get(repo, 0) + 1
            domain_stats[domain] = domain_stats.get(domain, 0) + 1

        summary = {
            "pipeline_summary": {
                "total_functions": total,
                "successful": success,
                "failed": failed,
                "success_rate": "{0:.2f}%".format((success/total*100) if total else 0),
                "run_timestamp": datetime.now().isoformat(),
                "model_used": self.model,
                "prompt_version": PROMPT_VERSION,
            },
            "repository_breakdown": repo_stats,
            "domain_breakdown": domain_stats,
        }

        summary_file = os.path.join(BATCH_DIR, "mock_annotation_summary.json")
        with open(summary_file, 'w') as fh:
            json.dump(summary, fh, indent=2)

        logger.info("Summary -> %s  |  Success: %s / %s (%.2f%%)",
                    summary_file, success, total, (success / total * 100) if total else 0)


def main():
    parser = argparse.ArgumentParser(
    description="Phase 6 - Generate mock natural-language requirements for concurrent Java functions"
    )
    parser.add_argument("--batch-size", type=int, default=50, help="Functions per batch")
    parser.add_argument("--delay", type=float, default=0.1, help="Seconds between operations")
    args = parser.parse_args()

    pipeline = MockLLMAnnotationPipeline(
        request_delay = args.delay,
    )
    pipeline.annotate_all(batch_size=args.batch_size)

if __name__ == "__main__":
    main()
