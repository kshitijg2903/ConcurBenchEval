#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM Annotation Pipeline - Phase 6
Generates natural-language requirements for concurrent Java functions
via Together AI. Saves per-batch files, a master file, and a run summary.

Usage:
python fixed_llm_annotation_pipeline.py --batch-size 10 --delay 1.5
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime

import requests

# Constants
PROMPT_VERSION = "v2-20250628"
DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"  # Changed to free DeepSeek model
API_URL = "https://api.together.xyz/v1/chat/completions"
MAX_SOURCE_LINES = 400
BATCH_DIR = "concurrent_analysis_output"
LOG_FILE = os.path.join(BATCH_DIR, "llm_annotation.log")

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
class LLMAnnotationPipeline:
    def __init__(
        self,
        api_key,
        model=DEFAULT_MODEL_ID,
        request_delay=1.0,
        max_retries=3,
        stream=False,
    ):
        self.api_key = api_key
        self.model = model
        self.request_delay = request_delay
        self.max_retries = max_retries
        self.stream = stream

        self.analysis_files = [
            "concurrent_analysis_output/repositories/JCTools_JCTools_analysis.json",
            "concurrent_analysis_output/repositories/peptos_traffic-shm_analysis.json",
            "concurrent_analysis_output/repositories/pramalhe_ConcurrencyFreaks_analysis.json",
            "concurrent_analysis_output/repositories/RobAustin_low-latency-primitive-concurrent-queues_analysis.json",
        ]

        self.output_file = os.path.join(BATCH_DIR, "annotated_functions.json")

    # Prompt helpers
    @staticmethod
    def truncate_code(code, max_lines=MAX_SOURCE_LINES):
        lines = code.splitlines()
        if len(lines) <= max_lines:
            return code
        keep = max_lines // 2
        return "\n".join(lines[:keep] + ["// ... truncated ..."] + lines[-keep:])

    def create_prompt(self, fn):
        src = self.truncate_code(fn.get("source_code", "N/A"))
        return """For the following Java function, generate concise natural-language requirements using the provided metadata:

**Function Metadata**
- `name`: {0}
- `signature`: {1}
- `file_path`: {2}
- `dependencies`: {3}
- `concurrency_patterns`: {4}
- `sync_primitives`: {5}
- `domain`: {6}
- `complexity_score`: {7}

**Source Code**
```java
{8}
```

Follow this format:

What the function does - brief technical summary

Input-Output - use :param ... and :return: lines

Dependencies - list what it relies on

Concurrency patterns - describe any mechanisms used

Generate the requirements now:""".format(
            fn.get('name', 'N/A'),
            fn.get('signature', 'N/A'),
            fn.get('file_path', 'N/A'),
            fn.get('dependencies', []),
            fn.get('concurrency_patterns', []),
            fn.get('sync_primitives', []),
            fn.get('domain', 'N/A'),
            fn.get('complexity_score', 0),
            src
        )

    # Together AI call
    def call_llm_api(self, prompt):
        headers = {
            "Authorization": "Bearer {0}".format(self.api_key),
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "temperature": 0.3,
            "top_p": 0.9,
            "stream": self.stream,
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(API_URL, headers=headers, json=payload, timeout=60, stream=self.stream)
                if resp.status_code == 200:
                    if self.stream:
                        content_chunks = []
                        for line in resp.iter_lines(decode_unicode=True):
                            if not line:  # skip keep-alive
                                continue
                            if line.startswith("data:"):
                                json_chunk = json.loads(line.replace("data:", "").strip())
                                delta = json_chunk["choices"][0]["delta"].get("content", "")
                                content_chunks.append(delta)
                        return "".join(content_chunks).strip()
                    return resp.json()["choices"][0]["message"]["content"].strip()

                if resp.status_code == 429:
                    sleep_for = attempt * 5
                    logger.warning("HTTP 429 (rate-limit). Sleeping %ss...", sleep_for)
                    time.sleep(sleep_for)
                    continue

                logger.error("API error %s - %s", resp.status_code, resp.text)

            except Exception as exc:
                logger.error("Request attempt %s failed: %s", attempt, exc)

            if attempt < self.max_retries:
                time.sleep(2 ** attempt)

        return None

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
        logger.info("=== Phase 6 - LLM Annotation (%s) ===", PROMPT_VERSION)

        # Build work-list
        all_functions = []
        for af in self.analysis_files:
            all_functions.extend(self.load_functions(af))
        if not all_functions:
            logger.error("No functions loaded - aborting.")
            return
            
        # Filter for only functions with tests
        functions = [f for f in all_functions if f.get("test_functions") or f.get("test_files")]
        logger.info("Total functions to process: %d (filtered from %d total functions - only those with tests)", 
                   len(functions), len(all_functions))

        if not functions:
            logger.error("No functions with tests found - aborting.")
            return

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
                prompt = self.create_prompt(fn)
                content = self.call_llm_api(prompt)

                ann = dict(fn)  # Copy the original function data
                ann.update({
                    "llm_annotation": content or "Annotation failed",
                    "annotation_error": content is None,
                    "annotation_timestamp": datetime.now().isoformat(),
                    "model_used": self.model,
                    "prompt_version": PROMPT_VERSION,
                })
                batch_annotated.append(ann)
                annotated.append(ann)
                time.sleep(self.request_delay)

            # save batch
            batch_file = os.path.join(BATCH_DIR, "annotated_{0}.json".format(batch_tag))
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

        summary_file = os.path.join(BATCH_DIR, "annotation_summary.json")
        with open(summary_file, 'w') as fh:
            json.dump(summary, fh, indent=2)

        logger.info("Summary -> %s  |  Success: %s / %s (%.2f%%)",
                    summary_file, success, total, (success / total * 100) if total else 0)


def main():
    parser = argparse.ArgumentParser(
    description="Phase 6 - Generate natural-language requirements for concurrent Java functions"
    )
    parser.add_argument("--batch-size", type=int, default=25, help="Functions per batch")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between API calls")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="Together model ID")
    parser.add_argument("--stream", action="store_true", help="Use streaming responses")
    args = parser.parse_args()
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        parser.error("Please set the TOGETHER_API_KEY environment variable")

    pipeline = LLMAnnotationPipeline(
        api_key=api_key,
        model=args.model,
        request_delay=args.delay,
        stream=args.stream,
    )
    pipeline.annotate_all(batch_size=args.batch_size)

if __name__ == "__main__":
    main()
