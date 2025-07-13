#!/usr/bin/env python3
"""
LLM Annotation Pipeline – Phase 6 (Hardened, with Summary)
Generates natural-language requirements for concurrent Java functions
via Together AI.  Saves per-batch files, a master file, and a run summary.

Usage (from a VS Code terminal):

1.  Create & activate a venv              →  python -m venv .venv && source .venv/bin/activate
2.  Install deps                          →  pip install requests
3.  Export your Together AI key           →  export TOGETHER_API_KEY="tgpk_live_xxx"
4.  Run                                   →  python phase6_annotate.py --batch-size 50 --delay 1.2 --stream
"""

from __future__ import annotations

import argparse, json, logging, os, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests                                   # pip install requests


# ───────────────────────────────────────────
#  Constants
# ───────────────────────────────────────────
PROMPT_VERSION     = "v2-20250626"
DEFAULT_MODEL_ID   = "meta-llama/Llama-3-70b-instruct"   # check Together ID
API_URL            = "https://api.together.xyz/v1/chat/completions"
MAX_SOURCE_LINES   = 400
BATCH_DIR          = Path("concurrent_analysis_output")
LOG_FILE           = BATCH_DIR / "llm_annotation.log"


# ───────────────────────────────────────────
#  Logging setup
# ───────────────────────────────────────────
BATCH_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ───────────────────────────────────────────
#  Pipeline class
# ───────────────────────────────────────────
class LLMAnnotationPipeline:
    def __init__(
        self,
        api_key: str,
        model: str          = DEFAULT_MODEL_ID,
        request_delay: float = 1.0,
        max_retries: int     = 3,
        stream: bool         = False,
    ) -> None:
        self.api_key       = api_key
        self.model         = model
        self.request_delay = request_delay
        self.max_retries   = max_retries
        self.stream        = stream

        self.analysis_files: List[str] = [
            "concurrent_analysis_output/repositories/JCTools_JCTools_analysis.json",
            "concurrent_analysis_output/repositories/peptos_traffic-shm_analysis.json",
            "concurrent_analysis_output/repositories/pramalhe_ConcurrencyFreaks_analysis.json",
            "concurrent_analysis_output/repositories/RobAustin_low-latency-primitive-concurrent-queues_analysis.json",
        ]

        self.output_file = BATCH_DIR / "annotated_functions.json"

    # ───────────────────────────────
    #  Prompt helpers
    # ───────────────────────────────
    @staticmethod
    def truncate_code(code: str, max_lines: int = MAX_SOURCE_LINES) -> str:
        lines = code.splitlines()
        if len(lines) <= max_lines:
            return code
        keep = max_lines // 2
        return "\n".join(lines[:keep] + ["// … truncated …"] + lines[-keep:])

    def create_prompt(self, fn: Dict[str, Any]) -> str:
        src = self.truncate_code(fn.get("source_code", "N/A"))
        return f"""For the following Java function, generate concise natural-language requirements using the provided metadata:

**Function Metadata**
- `name`: {fn.get('name', 'N/A')}
- `signature`: {fn.get('signature', 'N/A')}
- `file_path`: {fn.get('file_path', 'N/A')}
- `dependencies`: {fn.get('dependencies', [])}
- `concurrency_patterns`: {fn.get('concurrency_patterns', [])}
- `sync_primitives`: {fn.get('sync_primitives', [])}
- `domain`: {fn.get('domain', 'N/A')}
- `complexity_score`: {fn.get('complexity_score', 0)}

**Source Code**
```java
{src}

Follow this format:

What the function does – brief technical summary

Input-Output – use :param … and :return: lines

Dependencies – list what it relies on

Concurrency patterns – describe any mechanisms used

Generate the requirements now:"""
    
# ───────────────────────────────
#  Together AI call
# ───────────────────────────────
def call_llm_api(self, prompt: str) -> Optional[str]:
    headers = {
        "Authorization": f"Bearer {self.api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
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
                    content_chunks: List[str] = []
                    for line in resp.iter_lines(decode_unicode=True):
                        if not line:              # skip keep-alive
                            continue
                        if line.startswith(b"data:"):
                            json_chunk = json.loads(line.removeprefix(b"data:").strip())
                            delta = json_chunk["choices"][0]["delta"].get("content", "")
                            content_chunks.append(delta)
                    return "".join(content_chunks).strip()
                return resp.json()["choices"][0]["message"]["content"].strip()

            if resp.status_code == 429:
                sleep_for = attempt * 5
                logger.warning("HTTP 429 (rate-limit). Sleeping %ss…", sleep_for)
                time.sleep(sleep_for)
                continue

            logger.error("API error %s – %s", resp.status_code, resp.text)

        except Exception as exc:
            logger.error("Request attempt %s failed: %s", attempt, exc)

        if attempt < self.max_retries:
            time.sleep(2 ** attempt)

    return None

# ───────────────────────────────
#  Disk helpers
# ───────────────────────────────
@staticmethod
def load_functions(path: str) -> List[Dict[str, Any]]:
    fp = Path(path)
    if not fp.exists():
        logger.warning("File not found: %s", path)
        return []
    try:
        with fp.open(encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "functions" in data:
            return data["functions"]
        logger.error("Unexpected JSON structure in %s", path)
    except Exception as exc:
        logger.error("Failed reading %s – %s", path, exc)
    return []

# ───────────────────────────────
#  Main driver
# ───────────────────────────────
def annotate_all(self, batch_size: int = 50) -> None:
    logger.info("=== Phase 6 – LLM Annotation (%s) ===", PROMPT_VERSION)

    # Build work-list
    functions: List[Dict[str, Any]] = []
    for af in self.analysis_files:
        functions.extend(self.load_functions(af))
    if not functions:
        logger.error("No functions loaded – aborting.")
        return
    logger.info("Total functions to process: %d", len(functions))

    annotated: List[Dict[str, Any]] = []
    now_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for start in range(0, len(functions), batch_size):
        batch = functions[start:start + batch_size]
        batch_idx = start // batch_size + 1
        batch_tag = f"{now_stamp}_batch{batch_idx}"
        logger.info("⚙  Batch %s (%d functions)", batch_tag, len(batch))

        batch_annotated: List[Dict[str, Any]] = []
        for idx, fn in enumerate(batch, start=start + 1):
            logger.info("   · %d/%d  %s", idx, len(functions), fn.get("name"))
            prompt   = self.create_prompt(fn)
            content  = self.call_llm_api(prompt)

            ann = {
                **fn,
                "llm_annotation": content or "Annotation failed",
                "annotation_error": content is None,
                "annotation_timestamp": datetime.now(timezone.utc).isoformat(),
                "model_used": self.model,
                "prompt_version": PROMPT_VERSION,
            }
            batch_annotated.append(ann)
            annotated.append(ann)
            time.sleep(self.request_delay)

        # save batch
        batch_file = BATCH_DIR / f"annotated_{batch_tag}.json"
        with batch_file.open("w", encoding="utf-8") as fh:
            json.dump(batch_annotated, fh, indent=2, ensure_ascii=False)
        logger.info("   ✔ Saved %s", batch_file)

    # save full corpus
    with self.output_file.open("w", encoding="utf-8") as fh:
        json.dump(annotated, fh, indent=2, ensure_ascii=False)
    logger.info("★ All annotations written → %s", self.output_file)

    # summary
    self.generate_summary(annotated)

# ───────────────────────────────
#  Summary stats
# ───────────────────────────────
def generate_summary(self, annotated: List[Dict[str, Any]]) -> None:
    total   = len(annotated)
    failed  = sum(1 for f in annotated if f.get("annotation_error"))
    success = total - failed
    repo_stats: Dict[str, int]   = {}
    domain_stats: Dict[str, int] = {}

    for fn in annotated:
        repo   = fn.get("repository", "unknown")
        domain = fn.get("domain",     "unknown")
        repo_stats[repo]   = repo_stats.get(repo,   0) + 1
        domain_stats[domain] = domain_stats.get(domain, 0) + 1

    summary = {
        "pipeline_summary": {
            "total_functions"     : total,
            "successful"          : success,
            "failed"              : failed,
            "success_rate"        : f"{(success/total*100):.2f}%" if total else "0%",
            "run_timestamp"       : datetime.now(timezone.utc).isoformat(),
            "model_used"          : self.model,
            "prompt_version"      : PROMPT_VERSION,
        },
        "repository_breakdown": repo_stats,
        "domain_breakdown"    : domain_stats,
    }

    summary_file = BATCH_DIR / "annotation_summary.json"
    with summary_file.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    logger.info("Summary → %s  |  Success: %s / %s (%.2f%%)",
                summary_file, success, total, success / total * 100)


def main() -> None:
    parser = argparse.ArgumentParser(
    description="Phase 6 – Generate natural-language requirements for concurrent Java functions"
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
        api_key       = api_key,
        model         = args.model,
        request_delay = args.delay,
        stream        = args.stream,
    )
    pipeline.annotate_all(batch_size=args.batch_size)

if __name__ == "main":
    main()


# #!/usr/bin/env python3
# """
# LLM Annotation Pipeline - Phase 6
# Generates natural language requirements for concurrent Java functions using Together AI API
# """

# import json
# import os
# import time
# import requests
# from typing import Dict, List, Any
# from pathlib import Path
# import logging
# from datetime import datetime

# # Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('llm_annotation.log'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# class LLMAnnotationPipeline:
#     def __init__(self, api_key: str):
#         self.api_key = api_key
#         self.api_url = "https://api.together.xyz/v1/chat/completions"
#         self.model = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
        
#         # Repository analysis files
#         self.analysis_files = [
#             "concurrent_analysis_output/repositories/JCTools_JCTools_analysis.json",
#             "concurrent_analysis_output/repositories/peptos_traffic-shm_analysis.json", 
#             "concurrent_analysis_output/repositories/pramalhe_ConcurrencyFreaks_analysis.json",
#             "concurrent_analysis_output/repositories/RobAustin_low-latency-primitive-concurrent-queues_analysis.json"
#         ]
        
#         # Output file
#         self.output_file = "concurrent_analysis_output/annotated_functions.json"
        
#         # Rate limiting
#         self.request_delay = 1.0  # seconds between requests
#         self.max_retries = 3
        
#     def create_prompt(self, function_data: Dict[str, Any]) -> str:
#         """Create the annotation prompt for a function"""
        
#         prompt = f"""For the following Java function, generate concise natural language requirements using the provided metadata:

# **Function Metadata:**
# * `name`: {function_data.get('name', 'N/A')}
# * `signature`: {function_data.get('signature', 'N/A')}
# * `file_path`: {function_data.get('file_path', 'N/A')}
# * `dependencies`: {function_data.get('dependencies', [])}
# * `concurrency_patterns`: {function_data.get('concurrency_patterns', [])}
# * `sync_primitives`: {function_data.get('sync_primitives', [])}
# * `domain`: {function_data.get('domain', 'N/A')}
# * `complexity_score`: {function_data.get('complexity_score', 0)}
# * `repository`: {function_data.get('repository', 'N/A')}

# **Source Code:**
# ```java
# {function_data.get('source_code', 'N/A')}
# ```

# Follow this format for the response:

# 1. **What the function does** – Provide a short, clear technical summary.
# 2. **Input-Output** – `:param <param_name>: <type>, <description>` for each parameter, and `:return: <type>, <description>` or `No return values.`
# 3. **Dependencies** – List what the function depends on.
# 4. **Concurrency patterns** – Mention and explain any concurrency mechanisms used.

# Generate the requirements now:"""
        
#         return prompt
    
#     def call_llm_api(self, prompt: str) -> str:
#         """Make API call to Together AI"""
        
#         headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Content-Type": "application/json"
#         }
        
#         payload = {
#             "model": self.model,
#             "messages": [
#                 {
#                     "role": "user",
#                     "content": prompt
#                 }
#             ],
#             "max_tokens": 1000,
#             "temperature": 0.3,
#             "top_p": 0.9,
#             "stop": None
#         }
        
#         for attempt in range(self.max_retries):
#             try:
#                 response = requests.post(
#                     self.api_url,
#                     headers=headers,
#                     json=payload,
#                     timeout=30
#                 )
                
#                 if response.status_code == 200:
#                     result = response.json()
#                     return result['choices'][0]['message']['content']
#                 elif response.status_code == 429:  # Rate limit
#                     wait_time = (attempt + 1) * 5
#                     logger.warning(f"Rate limited, waiting {wait_time}s before retry {attempt + 1}")
#                     time.sleep(wait_time)
#                 else:
#                     logger.error(f"API error: {response.status_code} - {response.text}")
                    
#             except Exception as e:
#                 logger.error(f"Request failed (attempt {attempt + 1}): {str(e)}")
#                 if attempt < self.max_retries - 1:
#                     time.sleep(2 ** attempt)  # Exponential backoff
        
#         return "Failed to generate annotation"
    
#     def load_functions_from_file(self, file_path: str) -> List[Dict[str, Any]]:
#         """Load functions from a repository analysis file"""
        
#         if not os.path.exists(file_path):
#             logger.warning(f"File not found: {file_path}")
#             return []
            
#         try:
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 data = json.load(f)
                
#             if isinstance(data, list):
#                 functions = data
#             elif isinstance(data, dict) and 'functions' in data:
#                 functions = data['functions']
#             else:
#                 logger.error(f"Unexpected data structure in {file_path}")
#                 return []
                
#             logger.info(f"Loaded {len(functions)} functions from {file_path}")
#             return functions
            
#         except Exception as e:
#             logger.error(f"Error loading {file_path}: {str(e)}")
#             return []
    
#     def load_all_functions(self) -> List[Dict[str, Any]]:
#         """Load functions from all repository analysis files"""
        
#         all_functions = []
        
#         for file_path in self.analysis_files:
#             functions = self.load_functions_from_file(file_path)
#             all_functions.extend(functions)
        
#         logger.info(f"Total functions loaded: {len(all_functions)}")
#         return all_functions
    
#     def annotate_function(self, function_data: Dict[str, Any]) -> Dict[str, Any]:
#         """Annotate a single function with LLM-generated requirements"""
        
#         prompt = self.create_prompt(function_data)
#         annotation = self.call_llm_api(prompt)
        
#         # Create annotated function entry
#         annotated_function = {
#             **function_data,  # Copy all original metadata
#             "llm_annotation": annotation,
#             "annotation_timestamp": datetime.now().isoformat(),
#             "model_used": self.model
#         }
        
#         return annotated_function
    
#     def save_progress(self, annotated_functions: List[Dict[str, Any]], batch_num: int):
#         """Save progress to avoid losing work"""
        
#         progress_file = f"concurrent_analysis_output/annotated_functions_batch_{batch_num}.json"
#         os.makedirs(os.path.dirname(progress_file), exist_ok=True)
        
#         with open(progress_file, 'w', encoding='utf-8') as f:
#             json.dump(annotated_functions, f, indent=2, ensure_ascii=False)
        
#         logger.info(f"Saved progress: {len(annotated_functions)} functions to {progress_file}")
    
#     def run_annotation_pipeline(self, batch_size: int = 50):
#         """Run the complete annotation pipeline"""
        
#         logger.info("Starting LLM Annotation Pipeline - Phase 6")
        
#         # Load all functions
#         all_functions = self.load_all_functions()
        
#         if not all_functions:
#             logger.error("No functions loaded. Exiting.")
#             return
        
#         # Create output directory
#         os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
#         annotated_functions = []
#         batch_num = 0
        
#         # Process functions in batches
#         for i in range(0, len(all_functions), batch_size):
#             batch = all_functions[i:i + batch_size]
#             batch_num += 1
            
#             logger.info(f"Processing batch {batch_num} ({len(batch)} functions)")
            
#             batch_annotated = []
            
#             for j, function_data in enumerate(batch):
#                 try:
#                     logger.info(f"Annotating function {i + j + 1}/{len(all_functions)}: {function_data.get('name', 'unnamed')}")
                    
#                     annotated_function = self.annotate_function(function_data)
#                     batch_annotated.append(annotated_function)
#                     annotated_functions.append(annotated_function)
                    
#                     # Rate limiting
#                     time.sleep(self.request_delay)
                    
#                 except Exception as e:
#                     logger.error(f"Failed to annotate function {function_data.get('name', 'unnamed')}: {str(e)}")
#                     # Add function with error annotation
#                     error_function = {
#                         **function_data,
#                         "llm_annotation": f"Annotation failed: {str(e)}",
#                         "annotation_timestamp": datetime.now().isoformat(),
#                         "model_used": self.model,
#                         "annotation_error": True
#                     }
#                     batch_annotated.append(error_function)
#                     annotated_functions.append(error_function)
            
#             # Save batch progress
#             self.save_progress(batch_annotated, batch_num)
        
#         # Save final results
#         with open(self.output_file, 'w', encoding='utf-8') as f:
#             json.dump(annotated_functions, f, indent=2, ensure_ascii=False)
        
#         # Generate summary
#         self.generate_summary(annotated_functions)
        
#         logger.info(f"Pipeline completed! Annotated {len(annotated_functions)} functions")
#         logger.info(f"Results saved to: {self.output_file}")
    
#     def generate_summary(self, annotated_functions: List[Dict[str, Any]]):
#         """Generate summary statistics"""
        
#         total_functions = len(annotated_functions)
#         error_count = sum(1 for f in annotated_functions if f.get('annotation_error', False))
#         success_count = total_functions - error_count
        
#         # Repository breakdown
#         repo_stats = {}
#         for func in annotated_functions:
#             repo = func.get('repository', 'unknown')
#             repo_stats[repo] = repo_stats.get(repo, 0) + 1
        
#         # Domain breakdown
#         domain_stats = {}
#         for func in annotated_functions:
#             domain = func.get('domain', 'unknown')
#             domain_stats[domain] = domain_stats.get(domain, 0) + 1
        
#         summary = {
#             "pipeline_summary": {
#                 "total_functions_processed": total_functions,
#                 "successful_annotations": success_count,
#                 "failed_annotations": error_count,
#                 "success_rate": f"{(success_count/total_functions)*100:.2f}%" if total_functions > 0 else "0%",
#                 "processing_timestamp": datetime.now().isoformat(),
#                 "model_used": self.model
#             },
#             "repository_breakdown": repo_stats,
#             "domain_breakdown": domain_stats
#         }
        
#         summary_file = "concurrent_analysis_output/annotation_summary.json"
#         with open(summary_file, 'w', encoding='utf-8') as f:
#             json.dump(summary, f, indent=2, ensure_ascii=False)
        
#         logger.info("Summary Statistics:")
#         logger.info(f"  Total functions: {total_functions}")
#         logger.info(f"  Successful: {success_count}")
#         logger.info(f"  Failed: {error_count}")
#         logger.info(f"  Success rate: {(success_count/total_functions)*100:.2f}%")
#         logger.info(f"Summary saved to: {summary_file}")

# def main():
#     """Main execution function"""
    
#     # Together AI API key
#     API_KEY = "tgp_v1_yRhgJ_I2JHyxZN90uDcpILrgN-rDsffIgWsoASMcUoU"
    
#     # Initialize pipeline
#     pipeline = LLMAnnotationPipeline(API_KEY)
    
#     # Run annotation pipeline
#     pipeline.run_annotation_pipeline(batch_size=25)  # Smaller batches for stability

# if __name__ == "__main__":
#     main()



