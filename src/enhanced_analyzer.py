import json
import os
import re
import ast
import subprocess
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import shutil
from collections import defaultdict

@dataclass
class TestFunction:
    """Represents a test function"""
    name: str
    signature: str
    file_path: str
    source_code: str
    line_number: int
    tested_class: str = ""
    tested_methods: List[str] = None
    test_type: str = ""  # unit, integration, performance
    
    def __post_init__(self):
        if self.tested_methods is None:
            self.tested_methods = []

@dataclass
class ConcurrentFunction:
    """Represents a concurrent function with its properties"""
    name: str
    signature: str
    file_path: str
    language: str
    sync_primitives: List[str]
    dependencies: List[str]
    concurrency_patterns: List[str]
    complexity_score: int
    source_code: str
    line_number: int
    repository: str
    test_files: List[str] = None
    test_functions: List[TestFunction] = None
    domain: str = ""
    class_name: str = ""
    method_type: str = ""  # constructor, method, static_method
    
    def __post_init__(self):
        if self.test_files is None:
            self.test_files = []
        if self.test_functions is None:
            self.test_functions = []

class RefinedJavaConcurrentAnalyzer:
    def __init__(self, dataset_file: str = "concurrent_structures_dataset.json", 
                 clone_dir: str = "cloned_repositories"):
        """Initialize analyzer with collected dataset"""
        with open(dataset_file, 'r') as f:
            self.dataset = json.load(f)
        
        self.clone_dir = Path(clone_dir)
        self.clone_dir.mkdir(exist_ok=True)
        
        # Create organized output structure
        self.output_dir = Path("concurrent_analysis_output")
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "repositories").mkdir(exist_ok=True)
        (self.output_dir / "functions").mkdir(exist_ok=True)
        (self.output_dir / "tests").mkdir(exist_ok=True)
        
        # Enhanced Java concurrent patterns
        self.java_concurrent_imports = [
            'java.util.concurrent', 'java.util.concurrent.atomic', 'java.util.concurrent.locks',
            'java.lang.Thread', 'java.util.concurrent.ExecutorService', 'java.util.concurrent.Future',
            'java.util.concurrent.CompletableFuture', 'java.util.concurrent.ConcurrentHashMap',
            'java.util.concurrent.ConcurrentLinkedQueue', 'java.util.concurrent.BlockingQueue',
            'java.util.concurrent.Semaphore', 'java.util.concurrent.CountDownLatch',
            'java.util.concurrent.CyclicBarrier', 'java.util.concurrent.Phaser'
        ]
        
        self.java_concurrent_keywords = [
            # Synchronization
            'synchronized', 'volatile', 'wait', 'notify', 'notifyAll',
            # Atomic operations
            'AtomicInteger', 'AtomicLong', 'AtomicReference', 'AtomicBoolean',
            'AtomicIntegerArray', 'AtomicLongArray', 'AtomicReferenceArray',
            'compareAndSet', 'getAndSet', 'getAndIncrement', 'getAndDecrement',
            'getAndAdd', 'incrementAndGet', 'decrementAndGet', 'addAndGet',
            'compareAndExchange', 'weakCompareAndSet',
            # Locks
            'ReentrantLock', 'ReadWriteLock', 'StampedLock', 'Lock', 'Condition',
            'lock', 'unlock', 'tryLock', 'lockInterruptibly', 'newCondition',
            'await', 'signal', 'signalAll',
            # Concurrent collections
            'ConcurrentHashMap', 'ConcurrentLinkedQueue', 'ConcurrentLinkedDeque',
            'ConcurrentSkipListMap', 'ConcurrentSkipListSet', 'CopyOnWriteArrayList',
            'CopyOnWriteArraySet', 'BlockingQueue', 'ArrayBlockingQueue',
            'LinkedBlockingQueue', 'PriorityBlockingQueue', 'DelayQueue',
            'SynchronousQueue', 'LinkedTransferQueue', 'BlockingDeque',
            # Executors and threading
            'ExecutorService', 'ThreadPoolExecutor', 'ScheduledExecutorService',
            'ForkJoinPool', 'CompletableFuture', 'Future', 'Callable',
            'Runnable', 'Thread', 'ThreadLocal', 'InheritableThreadLocal',
            # Synchronization utilities
            'Semaphore', 'CountDownLatch', 'CyclicBarrier', 'Phaser', 'Exchanger',
            # Memory model
            'final', 'static', 'ThreadSafe', 'GuardedBy', 'Immutable'
        ]
        
        # Domain classification keywords
        self.domain_keywords = {
            'data_structures': ['queue', 'stack', 'list', 'map', 'hash', 'tree', 'graph', 
                               'collection', 'set', 'deque', 'array', 'buffer'],
            'synchronization': ['lock', 'mutex', 'semaphore', 'barrier', 'condition', 
                               'latch', 'synchronized', 'monitor', 'critical'],
            'memory_management': ['allocate', 'deallocate', 'pool', 'cache', 'buffer', 
                                'memory', 'heap', 'reference', 'weak', 'soft'],
            'thread_management': ['thread', 'worker', 'executor', 'scheduler', 'pool',
                                'fork', 'join', 'parallel', 'async', 'future'],
            'atomic_operations': ['atomic', 'cas', 'compare', 'swap', 'increment', 
                                'decrement', 'exchange', 'volatile'],
            'producer_consumer': ['producer', 'consumer', 'put', 'take', 'offer', 
                                'poll', 'blocking', 'queue'],
            'utilities': ['util', 'helper', 'factory', 'builder', 'adapter', 'wrapper']
        }

    def clone_selected_repositories(self, max_repos: int = 10) -> List[Dict]:
        """Clone top Java repositories for analysis"""
        print(f"\n📥 Cloning top {max_repos} Java repositories...")
        
        repositories = self.dataset['repositories']
        print(f"📊 Found {len(repositories)} repositories in dataset")
        
        # Filter Java repositories and sort by concurrent file count and stars
        java_repos = [repo for repo in repositories if repo.get('language') == 'Java']
        top_repos = sorted(java_repos, 
                          key=lambda x: (
                              x.get('structure_analysis', {}).get('concurrent_file_count', 0),
                              x.get('stargazers_count', 0)
                          ), 
                          reverse=True)[:max_repos]
        
        cloned_repos = []
        for repo in top_repos:
            repo_name = repo['name']
            
            # Handle both string and dict formats for owner
            if isinstance(repo['owner'], dict):
                owner = repo['owner']['login']
            else:
                owner = repo['owner']
                
            clone_url = repo['clone_url']
            repo_path = self.clone_dir / f"{owner}_{repo_name}"
            
            if repo_path.exists():
                print(f"  ✓ Repository {owner}/{repo_name} already exists")
            else:
                try:
                    print(f"  📦 Cloning {owner}/{repo_name}...")
                    subprocess.run(['git', 'clone', '--depth', '1', clone_url, str(repo_path)], 
                                 check=True, capture_output=True, text=True)
                    print(f"  ✓ Successfully cloned {owner}/{repo_name}")
                except subprocess.CalledProcessError as e:
                    print(f"  ❌ Failed to clone {owner}/{repo_name}: {e}")
                    continue
            
            repo['local_path'] = str(repo_path)
            cloned_repos.append(repo)
        
        print(f"📊 Successfully cloned {len(cloned_repos)} repositories")
        return cloned_repos

    def analyze_java_files(self, cloned_repos: List[Dict]) -> Dict[str, List[ConcurrentFunction]]:
        """Analyze all Java files in repositories including test discovery"""
        print("\n🔍 Analyzing Java concurrent functions and their tests...")
        
        repo_functions = {}
        
        for repo in cloned_repos:
            repo_name = f"{repo['owner'] if isinstance(repo['owner'], str) else repo['owner']['login']}/{repo['name']}"
            print(f"\n  📂 Analyzing {repo_name}...")
            
            repo_path = Path(repo['local_path'])
            functions = []
            
            # Find all Java files (both main and test)
            java_files = list(repo_path.rglob("*.java"))
            print(f"    📄 Found {len(java_files)} Java files total")
            
            # Separate main and test files
            main_files = []
            test_files = []
            
            for java_file in java_files:
                if self._is_test_file(java_file):
                    test_files.append(java_file)
                else:
                    main_files.append(java_file)
            
            print(f"    📄 Main files: {len(main_files)}, Test files: {len(test_files)}")
            
            # Process main files first
            concurrent_files = 0
            for java_file in main_files:
                try:
                    file_size = java_file.stat().st_size
                    if file_size > 500000:  # 500KB limit
                        continue
                        
                    with open(java_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Quick check if file contains concurrent code
                    if not self._has_concurrent_code(content):
                        continue
                    
                    concurrent_files += 1
                    file_functions = self.extract_java_concurrent_functions(
                        content, str(java_file), repo_name
                    )
                    functions.extend(file_functions)
                    
                except Exception as e:
                    print(f"    ⚠️  Error analyzing {java_file.name}: {e}")
                    continue
            
            print(f"    ✓ Analyzed {concurrent_files} concurrent files")
            print(f"    ✓ Found {len(functions)} concurrent functions")
            
            # Now find and link test files
            self._link_test_files(functions, test_files, repo_path)
            
            repo_functions[repo_name] = functions
        
        return repo_functions

    def _is_test_file(self, file_path: Path) -> bool:
        """Enhanced test file detection"""
        path_str = str(file_path).lower()
        file_name = file_path.name.lower()
        
        # Check if file is in test directories
        path_parts = [part.lower() for part in file_path.parts]
        test_dir_indicators = ['test', 'tests', 'testing', 'junit', 'spec', 'specs']
        
        # Check for test directory in path
        for part in path_parts:
            if any(indicator in part for indicator in test_dir_indicators):
                return True
        
        # Check filename patterns
        test_file_patterns = [
            r'test\.java$',           # Test.java
            r'.*test\.java$',         # XxxTest.java
            r'.*tests\.java$',        # XxxTests.java
            r'test.*\.java$',         # TestXxx.java
            r'.*spec\.java$',         # XxxSpec.java
            r'.*it\.java$',           # XxxIT.java (integration tests)
            r'.*benchmark\.java$',    # XxxBenchmark.java
        ]
        
        for pattern in test_file_patterns:
            if re.search(pattern, file_name):
                return True
        
        return False

    def _link_test_files(self, functions: List[ConcurrentFunction], test_files: List[Path], repo_path: Path):
        """Link test files to concurrent functions"""
        print(f"    🔗 Linking {len(test_files)} test files to functions...")
        
        test_functions_map = {}  # Maps class names to test functions
        
        # Process all test files
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Extract test functions from this file
                test_functions = self._extract_test_functions(content, str(test_file))
                
                # Get the class being tested
                tested_class = self._infer_tested_class(test_file, content)
                
                if tested_class:
                    if tested_class not in test_functions_map:
                        test_functions_map[tested_class] = []
                    test_functions_map[tested_class].extend(test_functions)
                
            except Exception as e:
                print(f"      ⚠️  Error processing test file {test_file.name}: {e}")
                continue
        
        # Link test functions to concurrent functions
        linked_count = 0
        for func in functions:
            class_name = func.class_name
            
            # Try exact match first
            if class_name in test_functions_map:
                func.test_functions = test_functions_map[class_name]
                func.test_files = list(set([tf.file_path for tf in func.test_functions]))
                linked_count += 1
            else:
                # Try partial matches
                for test_class, test_funcs in test_functions_map.items():
                    if (class_name.lower() in test_class.lower() or 
                        test_class.lower().replace('test', '') == class_name.lower()):
                        func.test_functions.extend(test_funcs)
                        func.test_files.extend([tf.file_path for tf in test_funcs])
                        
                if func.test_functions:
                    func.test_files = list(set(func.test_files))
                    linked_count += 1
        
        print(f"    ✓ Linked tests to {linked_count}/{len(functions)} functions")

    def _extract_test_functions(self, content: str, file_path: str) -> List[TestFunction]:
        """Extract test functions from Java test file"""
        test_functions = []
        lines = content.split('\n')
        
        # Look for test methods (methods annotated with @Test or containing test in name)
        test_patterns = [
            r'@Test',
            r'@ParameterizedTest',
            r'@RepeatedTest',
            r'@TestFactory',
            r'@DisplayName'
        ]
        
        method_pattern = r'((?:public|private|protected|static|final|\s)+)\s+(\w+(?:<[^>]*>)?(?:\[\])*)\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+[^{]+)?\s*\{'
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Check for test annotations or test method names
            is_test_method = False
            for pattern in test_patterns:
                if re.search(pattern, line):
                    is_test_method = True
                    break
            
            # Also check if method name contains 'test'
            if not is_test_method and 'test' in line.lower():
                method_match = re.search(method_pattern, line)
                if method_match:
                    method_name = method_match.group(3)
                    if 'test' in method_name.lower():
                        is_test_method = True
            
            if is_test_method:
                # Look for the actual method definition (might be on next lines)
                method_line_idx = i
                while method_line_idx < len(lines):
                    method_match = re.search(method_pattern, lines[method_line_idx])
                    if method_match:
                        method_name = method_match.group(3)
                        method_body = self._extract_method_body(lines, method_line_idx, max_lines=100)
                        
                        # Infer tested methods from the test code
                        tested_methods = self._infer_tested_methods(method_body)
                        
                        test_func = TestFunction(
                            name=method_name,
                            signature=lines[method_line_idx].strip(),
                            file_path=file_path,
                            source_code=method_body[:1000],  # Limit length
                            line_number=method_line_idx + 1,
                            tested_methods=tested_methods,
                            test_type=self._classify_test_type(method_name, method_body)
                        )
                        test_functions.append(test_func)
                        break
                    method_line_idx += 1
                    if method_line_idx - i > 5:  # Don't look too far
                        break
            
            i += 1
        
        return test_functions

    def _infer_tested_class(self, test_file_path: Path, content: str) -> str:
        """Infer which class is being tested"""
        # Method 1: Extract from filename
        file_name = test_file_path.stem
        
        # Remove common test suffixes
        suffixes_to_remove = ['Test', 'Tests', 'IT', 'Spec', 'Benchmark']
        for suffix in suffixes_to_remove:
            if file_name.endswith(suffix):
                return file_name[:-len(suffix)]
        
        # Remove test prefixes
        prefixes_to_remove = ['Test']
        for prefix in prefixes_to_remove:
            if file_name.startswith(prefix):
                return file_name[len(prefix):]
        
        # Method 2: Look for class imports or references in content
        import_matches = re.findall(r'import\s+(?:static\s+)?[\w.]+\.(\w+);', content)
        for imported_class in import_matches:
            if not imported_class.startswith('Test') and imported_class not in ['Assert', 'Assertions']:
                return imported_class
        
        return file_name

    def _infer_tested_methods(self, test_code: str) -> List[str]:
        """Infer which methods are being tested from test code"""
        tested_methods = []
        
        # Look for method calls that are likely being tested
        method_call_patterns = [
            r'\.(\w+)\s*\(',           # Direct method calls
            r'(\w+)\s*\(',             # Direct function calls
            r'new\s+\w+\([^)]*\)\.(\w+)\s*\(',  # Chained method calls
        ]
        
        for pattern in method_call_patterns:
            matches = re.findall(pattern, test_code)
            for match in matches:
                if isinstance(match, tuple):
                    method_name = match[-1]  # Get last group
                else:
                    method_name = match
                
                # Filter out common test methods and Java built-ins
                excluded_methods = {
                    'assertEquals', 'assertTrue', 'assertFalse', 'assertNull', 'assertNotNull',
                    'fail', 'setUp', 'tearDown', 'before', 'after', 'get', 'set', 'toString',
                    'equals', 'hashCode', 'wait', 'notify', 'notifyAll', 'size', 'isEmpty'
                }
                
                if (method_name not in excluded_methods and 
                    not method_name.startswith('assert') and
                    len(method_name) > 2 and
                    method_name[0].islower()):
                    tested_methods.append(method_name)
        
        return list(set(tested_methods))

    def _classify_test_type(self, method_name: str, test_code: str) -> str:
        """Classify the type of test"""
        method_name_lower = method_name.lower()
        code_lower = test_code.lower()
        
        if any(keyword in method_name_lower for keyword in ['performance', 'benchmark', 'stress']):
            return 'performance'
        elif any(keyword in method_name_lower for keyword in ['integration', 'end2end', 'e2e']):
            return 'integration'
        elif any(keyword in code_lower for keyword in ['thread', 'concurrent', 'parallel', 'executor']):
            return 'concurrency'
        else:
            return 'unit'

    # Keep all existing methods from the original class
    def extract_java_concurrent_functions(self, content: str, file_path: str, repo_name: str) -> List[ConcurrentFunction]:
        """Extract concurrent functions from Java file content"""
        functions = []
        lines = content.split('\n')
        
        # Get class name
        class_name = self._extract_class_name(content)
        
        # Find all method definitions with improved patterns
        method_patterns = [
            # Standard method pattern
            r'((?:public|private|protected|static|final|abstract|synchronized|native|\s)+)\s+(\w+(?:<[^>]*>)?(?:\[\])*)\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+[^{]+)?\s*\{',
            # Constructor pattern  
            r'((?:public|private|protected|\s)+)\s+(' + re.escape(class_name) + r')\s*\([^)]*\)\s*(?:throws\s+[^{]+)?\s*\{',
            # Method with generic return type
            r'((?:public|private|protected|static|final|synchronized|\s)+)<([^>]+)>\s+(\w+(?:\[\])*)\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+[^{]+)?\s*\{'
        ]
        
        for i, line in enumerate(lines):
            for pattern_idx, pattern in enumerate(method_patterns):
                match = re.search(pattern, line)
                if match:
                    # Extract method details based on pattern
                    if pattern_idx == 0:  # Standard method
                        modifiers = match.group(1).strip()
                        return_type = match.group(2)
                        method_name = match.group(3)
                        method_type = "constructor" if method_name == class_name else "method"
                    elif pattern_idx == 1:  # Constructor
                        modifiers = match.group(1).strip()
                        method_name = match.group(2)
                        return_type = "void"
                        method_type = "constructor"
                    else:  # Generic method
                        modifiers = match.group(1).strip()
                        generic_type = match.group(2)
                        return_type = match.group(3)
                        method_name = match.group(4)
                        method_type = "method"
                    
                    # Skip common non-methods
                    if method_name.lower() in ['if', 'while', 'for', 'switch', 'try', 'catch', 'else']:
                        continue
                    
                    # Extract method body
                    method_body = self._extract_method_body(lines, i)
                    full_method = line + '\n' + method_body
                    
                    # Check for concurrent elements
                    sync_primitives = self._find_concurrent_patterns(full_method)
                    concurrency_patterns = self._detect_high_level_patterns(full_method)
                    
                    # Only include methods with concurrent elements
                    if sync_primitives or concurrency_patterns or 'synchronized' in modifiers:
                        func = ConcurrentFunction(
                            name=method_name,
                            signature=line.strip(),
                            file_path=file_path,
                            language='java',
                            sync_primitives=sync_primitives,
                            dependencies=self._extract_java_dependencies(full_method),
                            concurrency_patterns=concurrency_patterns,
                            complexity_score=self._calculate_complexity_score(full_method),
                            source_code=method_body[:2000],  # Limit source code length
                            line_number=i + 1,
                            repository=repo_name,
                            domain=self._classify_function_domain(method_name, full_method),
                            class_name=class_name,
                            method_type=method_type
                        )
                        functions.append(func)
                    break
        
        return functions

    def _extract_class_name(self, content: str) -> str:
        """Extract class name from Java file"""
        class_match = re.search(r'(?:public\s+)?(?:abstract\s+)?(?:final\s+)?class\s+(\w+)', content)
        if class_match:
            return class_match.group(1)
        
        interface_match = re.search(r'(?:public\s+)?interface\s+(\w+)', content)
        if interface_match:
            return interface_match.group(1)
        
        enum_match = re.search(r'(?:public\s+)?enum\s+(\w+)', content)
        if enum_match:
            return enum_match.group(1)
        
        return "Unknown"

    def _has_concurrent_code(self, content: str) -> bool:
        """Check if Java file contains concurrent code"""
        content_lower = content.lower()
        
        # Check imports
        for import_pattern in self.java_concurrent_imports:
            if import_pattern.lower() in content_lower:
                return True
        
        # Check keywords
        for keyword in self.java_concurrent_keywords:
            if keyword.lower() in content_lower:
                return True
        
        return False

    def _find_concurrent_patterns(self, code: str) -> List[str]:
        """Find concurrent patterns in Java code"""
        found_patterns = []
        code_lower = code.lower()
        
        for keyword in self.java_concurrent_keywords:
            if keyword.lower() in code_lower:
                found_patterns.append(keyword)
        
        # Check for method calls that indicate concurrency
        concurrent_method_patterns = [
            r'\.compareAndSet\s*\(',
            r'\.getAndSet\s*\(',
            r'\.lock\s*\(',
            r'\.unlock\s*\(',
            r'\.await\s*\(',
            r'\.signal\s*\(',
            r'\.wait\s*\(',
            r'\.notify\s*\(',
            r'\.submit\s*\(',
            r'\.execute\s*\(',
            r'\.acquire\s*\(',
            r'\.release\s*\('
        ]
        
        for pattern in concurrent_method_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                method_name = pattern.replace(r'\.', '').replace(r'\s*\(', '')
                found_patterns.append(method_name)
        
        return list(set(found_patterns))

    def _detect_high_level_patterns(self, code: str) -> List[str]:
        """Detect high-level concurrency patterns"""
        patterns = []
        code_lower = code.lower()
        
        pattern_rules = {
            'double_checked_locking': ['if', 'null', 'synchronized', 'volatile'],
            'compare_and_swap': ['compareandset', 'atomic'],
            'producer_consumer': ['blockingqueue', 'put', 'take'],
            'reader_writer_lock': ['readwritelock', 'readlock', 'writelock'],
            'thread_pool': ['executorservice', 'threadpool', 'submit'],
            'future_pattern': ['future', 'completablefuture', 'get'],
            'observer_pattern': ['notify', 'observer', 'listener'],
            'singleton_pattern': ['synchronized', 'instance', 'static'],
            'barrier_synchronization': ['cyclicbarrier', 'countdownlatch', 'await'],
            'lock_free_algorithm': ['atomic', 'cas', 'compareandset', 'volatile'],
            'immutable_object': ['final', 'immutable', 'unmodifiable'],
            'thread_local_storage': ['threadlocal', 'get', 'set']
        }
        
        for pattern_name, keywords in pattern_rules.items():
            if sum(1 for keyword in keywords if keyword in code_lower) >= len(keywords) // 2:
                patterns.append(pattern_name)
        
        return patterns

    def _extract_java_dependencies(self, code: str) -> List[str]:
        """Extract method and class dependencies from Java code"""
        dependencies = []
        
        # Method calls
        method_calls = re.findall(r'\.(\w+)\s*\(', code)
        dependencies.extend(method_calls)
        
        # Static method calls
        static_calls = re.findall(r'(\w+)\.(\w+)\s*\(', code)
        dependencies.extend([f"{cls}.{method}" for cls, method in static_calls])
        
        # Class instantiations
        new_objects = re.findall(r'new\s+(\w+)(?:<[^>]*>)?\s*\(', code)
        dependencies.extend(new_objects)
        
        # Generic type usage
        generic_types = re.findall(r'<(\w+)(?:,\s*\w+)*>', code)
        dependencies.extend(generic_types)
        
        return list(set(dependencies))

    def _calculate_complexity_score(self, code: str) -> int:
        """Calculate complexity score for Java concurrent code"""
        score = 0
        code_lower = code.lower()
        
        # Concurrency complexity weights
        complexity_weights = {
            'synchronized': 5,
            'volatile': 3,
            'atomic': 4,
            'lock': 4,
            'condition': 4,
            'semaphore': 4,
            'barrier': 5,
            'latch': 3,
            'executor': 3,
            'future': 2,
            'concurrent': 2,
            'thread': 2
        }

        for keyword, weight in complexity_weights.items():
            score += code_lower.count(keyword) * weight
        
        # Control flow complexity
        score += code_lower.count('if') * 1
        score += code_lower.count('while') * 2
        score += code_lower.count('for') * 1
        score += code_lower.count('try') * 2
        score += code_lower.count('catch') * 1
        score += code_lower.count('finally') * 1
        
        # Nested complexity
        score += code.count('{') * 0.5  # Nesting penalty
        
        return int(score)

    def _classify_function_domain(self, func_name: str, code: str) -> str:
        """Classify function domain based on name and code content"""
        func_name_lower = func_name.lower()
        code_lower = code.lower()
        
        # Score each domain
        domain_scores = {}
        for domain, keywords in self.domain_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in func_name_lower:
                    score += 3
                if keyword in code_lower:
                    score += 1
            domain_scores[domain] = score
        
        # Return domain with highest score
        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            if domain_scores[best_domain] > 0:
                return best_domain
        
        return "general"

    def _extract_method_body(self, lines: List[str], start_line: int, max_lines: int = 200) -> str:
        """Extract method body with proper brace matching"""
        body_lines = []
        brace_count = 0
        started = False
        
        for i in range(start_line, min(len(lines), start_line + max_lines)):
            line = lines[i]
            
            # Skip the method signature line for the body
            if i > start_line:
                body_lines.append(line)
            
            # Count braces
            for char in line:
                if char == '{':
                    brace_count += 1
                    started = True
                elif char == '}':
                    brace_count -= 1
                    if started and brace_count == 0:
                        return '\n'.join(body_lines)
        
        return '\n'.join(body_lines)

    def _improved_test_detection(self, file_path: Path) -> bool:
        """Improved test file detection with multiple strategies"""
        path_str = str(file_path).lower()
        file_name = file_path.name.lower()
        
        # Strategy 1: Check directory structure
        path_parts = [part.lower() for part in file_path.parts]
        test_indicators = [
            'test', 'tests', 'testing', 'junit', 'spec', 'specs', 
            'it', 'integration', 'unit', 'e2e', 'acceptance'
        ]
        
        # Look for test directories in path
        for part in path_parts:
            if any(indicator in part for indicator in test_indicators):
                return True
        
        # Strategy 2: Check common test directory patterns
        test_path_patterns = [
            r'/test[s]?/',
            r'/src/test/',
            r'/src/it/',
            r'/testing/',
            r'/junit/',
            r'/spec[s]?/',
            r'\\test[s]?\\',
            r'\\src\\test\\',
            r'\\testing\\',
        ]
        
        for pattern in test_path_patterns:
            if re.search(pattern, path_str):
                return True
        
        # Strategy 3: Check filename patterns
        test_filename_patterns = [
            r'^test.*\.java$',           # TestXxx.java
            r'.*test\.java$',            # XxxTest.java
            r'.*tests\.java$',           # XxxTests.java
            r'.*it\.java$',              # XxxIT.java (integration tests)
            r'.*spec\.java$',            # XxxSpec.java
            r'.*benchmark\.java$',       # XxxBenchmark.java
            r'.*example\.java$',         # XxxExample.java
            r'.*demo\.java$',            # XxxDemo.java
        ]
        
        for pattern in test_filename_patterns:
            if re.match(pattern, file_name):
                return True
        
        # Strategy 4: Check file content if file is small enough
        try:
            if file_path.stat().st_size < 50000:  # Less than 50KB
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content_sample = f.read(2000)  # Read first 2KB
                    
                # Look for test annotations and imports
                test_content_indicators = [
                    '@Test', '@Before', '@After', '@BeforeEach', '@AfterEach',
                    '@BeforeClass', '@AfterClass', '@ParameterizedTest',
                    'import org.junit', 'import junit', 'import org.testng',
                    'import static org.junit', 'import static org.mockito',
                    'extends TestCase', 'extends TestSupport'
                ]
                
                content_lower = content_sample.lower()
                for indicator in test_content_indicators:
                    if indicator.lower() in content_lower:
                        return True
        except:
            pass  # If file can't be read, continue with other strategies
        
        return False

    def _enhanced_test_function_extraction(self, content: str, file_path: str) -> List[TestFunction]:
        """Enhanced test function extraction with better pattern matching"""
        test_functions = []
        lines = content.split('\n')
        
        # Comprehensive test method detection patterns
        test_indicators = {
            'annotations': [
                r'@Test(?:\([^)]*\))?',
                r'@ParameterizedTest',
                r'@RepeatedTest(?:\([^)]*\))?',
                r'@TestFactory',
                r'@TestTemplate',
                r'@DisplayName(?:\([^)]*\))?',
                r'@Timeout(?:\([^)]*\))?',
                r'@EnabledOn.*',
                r'@DisabledOn.*',
            ],
            'method_names': [
                r'test\w*',
                r'should\w*',
                r'verify\w*',
                r'check\w*',
                r'validate\w*',
                r'ensure\w*',
                r'given\w*When\w*Then\w*',
            ]
        }
        
        # Method signature pattern
        method_pattern = r'((?:public|private|protected|static|final|\s)+)\s+(\w+(?:<[^>]*>)?(?:\[\])*)\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+[^{]+)?\s*\{'
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Check for test method indicators
            is_test_method = False
            test_type = "unit"
            
            # Look back a few lines for annotations
            annotation_start = max(0, i - 5)
            annotation_context = '\n'.join(lines[annotation_start:i+1])
            
            # Check for test annotations
            for annotation_pattern in test_indicators['annotations']:
                if re.search(annotation_pattern, annotation_context, re.IGNORECASE):
                    is_test_method = True
                    break
            
            # Check for test method names
            if not is_test_method:
                for name_pattern in test_indicators['method_names']:
                    if re.search(f'\\b{name_pattern}\\b', line, re.IGNORECASE):
                        method_match = re.search(method_pattern, line, re.IGNORECASE)
                        if method_match:
                            is_test_method = True
                            break
            
            # If this looks like a test method, extract it
            if is_test_method:
                # Find the actual method definition
                method_line_idx = i
                while method_line_idx < len(lines) and method_line_idx < i + 3:
                    method_match = re.search(method_pattern, lines[method_line_idx], re.IGNORECASE)
                    if method_match:
                        modifiers = method_match.group(1).strip()
                        return_type = method_match.group(2)
                        method_name = method_match.group(3)
                        
                        # Extract method body with proper brace matching
                        method_body = self._extract_method_body(lines, method_line_idx, max_lines=150)
                        
                        # Classify test type
                        test_type = self._enhanced_test_classification(method_name, method_body, annotation_context)
                        
                        # Extract tested methods and classes
                        tested_methods = self._extract_tested_methods(method_body)
                        tested_class = self._extract_tested_class_from_test(method_body, file_path)
                        
                        test_func = TestFunction(
                            name=method_name,
                            signature=lines[method_line_idx].strip(),
                            file_path=file_path,
                            source_code=method_body[:1500],  # Limit length
                            line_number=method_line_idx + 1,
                            tested_class=tested_class,
                            tested_methods=tested_methods,
                            test_type=test_type
                        )
                        test_functions.append(test_func)
                        break
                    method_line_idx += 1
            
            i += 1
        
        return test_functions

    def _enhanced_test_classification(self, method_name: str, method_body: str, annotation_context: str) -> str:
        """Enhanced test classification with multiple strategies"""
        method_name_lower = method_name.lower()
        body_lower = method_body.lower()
        annotation_lower = annotation_context.lower()
        
        # Performance/Benchmark tests
        performance_indicators = [
            'performance', 'benchmark', 'stress', 'load', 'throughput',
            'latency', 'speed', 'timing', 'measure', 'profile'
        ]
        
        if any(indicator in method_name_lower for indicator in performance_indicators):
            return 'performance'
        
        if any(indicator in body_lower for indicator in performance_indicators):
            return 'performance'
        
        # Integration tests
        integration_indicators = [
            'integration', 'end2end', 'e2e', 'system', 'acceptance',
            'functional', 'scenario', 'workflow', 'pipeline'
        ]
        
        if any(indicator in method_name_lower for indicator in integration_indicators):
            return 'integration'
        
        # Concurrency tests
        concurrency_indicators = [
            'thread', 'concurrent', 'parallel', 'async', 'sync',
            'lock', 'atomic', 'volatile', 'executor', 'future',
            'race', 'deadlock', 'livelock'
        ]
        
        concurrency_count = sum(1 for indicator in concurrency_indicators 
                               if indicator in body_lower)
        if concurrency_count >= 2:
            return 'concurrency'
        
        # Mock/Stub tests
        if any(indicator in body_lower for indicator in ['mock', 'stub', 'spy', 'verify', 'when']):
            return 'mock'
        
        # Exception tests
        if any(indicator in body_lower for indicator in ['exception', 'throw', 'error', 'fail']):
            return 'exception'
        
        return 'unit'

    def _extract_tested_methods(self, test_code: str) -> List[str]:
        """Enhanced extraction of tested methods from test code"""
        tested_methods = set()
        
        # Pattern 1: Direct method calls (object.method())
        direct_calls = re.findall(r'(\w+)\.(\w+)\s*\(', test_code)
        for obj, method in direct_calls:
            if not self._is_test_utility_method(method):
                tested_methods.add(method)
        
        # Pattern 2: Static method calls (Class.method())
        static_calls = re.findall(r'([A-Z]\w*)\.(\w+)\s*\(', test_code)
        for cls, method in static_calls:
            if not self._is_test_utility_method(method):
                tested_methods.add(method)
        
        # Pattern 3: Method calls in assertions
        assertion_calls = re.findall(r'assert\w*\([^)]*\.(\w+)\s*\([^)]*\)', test_code)
        tested_methods.update(assertion_calls)
        
        # Pattern 4: Constructor calls (new Class())
        constructor_calls = re.findall(r'new\s+(\w+)\s*\(', test_code)
        tested_methods.update(constructor_calls)
        
        # Pattern 5: Method references (Class::method)
        method_refs = re.findall(r'\w+::(\w+)', test_code)
        tested_methods.update(method_refs)
        
        # Filter out common test utilities and Java built-ins
        filtered_methods = []
        for method in tested_methods:
            if not self._is_test_utility_method(method) and len(method) > 1:
                filtered_methods.append(method)
        
        return filtered_methods

    def _is_test_utility_method(self, method_name: str) -> bool:
        """Check if method is a test utility method that should be filtered out"""
        test_utility_methods = {
            # JUnit assertions
            'assertEquals', 'assertNotEquals', 'assertTrue', 'assertFalse',
            'assertNull', 'assertNotNull', 'assertSame', 'assertNotSame',
            'assertThat', 'assertThrows', 'assertDoesNotThrow', 'assertAll',
            'assertArrayEquals', 'assertIterableEquals', 'assertLinesMatch',
            'assertTimeout', 'assertTimeoutPreemptively', 'fail',
            
            # Mockito methods
            'when', 'thenReturn', 'thenThrow', 'verify', 'mock', 'spy',
            'doReturn', 'doThrow', 'doNothing', 'doAnswer', 'doCallRealMethod',
            'times', 'never', 'atLeast', 'atMost', 'only', 'inOrder',
            
            # Common test setup/teardown
            'setUp', 'tearDown', 'before', 'after', 'beforeEach', 'afterEach',
            'beforeAll', 'afterAll', 'given', 'givenThat',
            
            # Java built-ins
            'get', 'set', 'toString', 'equals', 'hashCode', 'wait', 'notify',
            'notifyAll', 'size', 'isEmpty', 'contains', 'add', 'remove',
            'clear', 'iterator', 'toArray', 'length', 'clone',
            
            # Test utilities
            'sleep', 'println', 'print', 'format', 'valueOf', 'parse'
        }
        
        return method_name in test_utility_methods or method_name.startswith('assert')

    def _extract_tested_class_from_test(self, test_code: str, test_file_path: str) -> str:
        """Extract the class being tested from test code and file path"""
        # Strategy 1: Extract from test file name
        test_file = Path(test_file_path)
        file_name = test_file.stem
        
        # Remove common test suffixes
        suffixes = ['Test', 'Tests', 'IT', 'Spec', 'Specs', 'Benchmark', 'Example', 'Demo']
        for suffix in suffixes:
            if file_name.endswith(suffix):
                potential_class = file_name[:-len(suffix)]
                if potential_class:
                    return potential_class
        
        # Remove common test prefixes
        prefixes = ['Test', 'IT']
        for prefix in prefixes:
            if file_name.startswith(prefix):
                potential_class = file_name[len(prefix):]
                if potential_class:
                    return potential_class
        
        # Strategy 2: Look for class instantiations in test code
        class_instantiations = re.findall(r'new\s+([A-Z]\w+)(?:<[^>]*>)?\s*\(', test_code)
        if class_instantiations:
            # Return the most common class instantiation
            from collections import Counter
            most_common = Counter(class_instantiations).most_common(1)
            if most_common:
                return most_common[0][0]
        
        # Strategy 3: Look for import statements or class references
        class_references = re.findall(r'([A-Z]\w+)\.(?:class|[A-Z_]+)', test_code)
        if class_references:
            return class_references[0]
        
        return file_name

    def _smart_test_to_function_matching(self, functions: List[ConcurrentFunction], 
                                       test_files: List[Path], repo_path: Path):
        """Enhanced test-to-function matching with multiple strategies"""
        print(f"    🔗 Smart matching {len(test_files)} test files to {len(functions)} functions...")
        
        # Build test function database
        test_database = {}  # Maps class names to test functions
        class_to_file_map = {}  # Maps class names to test file paths
        
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Extract test functions
                test_functions = self._enhanced_test_function_extraction(content, str(test_file))
                
                if test_functions:
                    # Get the primary tested class
                    tested_class = self._extract_tested_class_from_test(content, str(test_file))
                    
                    if tested_class not in test_database:
                        test_database[tested_class] = []
                        class_to_file_map[tested_class] = []
                    
                    test_database[tested_class].extend(test_functions)
                    class_to_file_map[tested_class].append(str(test_file))
                
            except Exception as e:
                print(f"      ⚠️  Error processing test file {test_file.name}: {e}")
                continue
        
        # Matching strategies
        matched_functions = 0
        
        for func in functions:
            class_name = func.class_name
            
            # Strategy 1: Exact class name match
            if class_name in test_database:
                func.test_functions = test_database[class_name]
                func.test_files = list(set(class_to_file_map[class_name]))
                matched_functions += 1
                continue
            
            # Strategy 2: Partial class name matching
            partial_matches = []
            for test_class in test_database.keys():
                if self._classes_match(class_name, test_class):
                    partial_matches.extend(test_database[test_class])
                    func.test_files.extend(class_to_file_map[test_class])
            
            if partial_matches:
                func.test_functions = partial_matches
                func.test_files = list(set(func.test_files))
                matched_functions += 1
                continue
            
            # Strategy 3: Method-based matching
            method_matches = []
            for test_class, test_funcs in test_database.items():
                for test_func in test_funcs:
                    if func.name.lower() in [m.lower() for m in test_func.tested_methods]:
                        method_matches.append(test_func)
                        func.test_files.append(test_func.file_path)
            
            if method_matches:
                func.test_functions = method_matches
                func.test_files = list(set(func.test_files))
                matched_functions += 1
        
        print(f"    ✓ Successfully matched tests to {matched_functions}/{len(functions)} functions")
        
        # Print matching statistics
        test_type_stats = {}
        for func in functions:
            if func.test_functions:
                test_types = [tf.test_type for tf in func.test_functions]
                for test_type in test_types:
                    test_type_stats[test_type] = test_type_stats.get(test_type, 0) + 1
        
        if test_type_stats:
            print(f"    📊 Test types found: {dict(sorted(test_type_stats.items(), key=lambda x: x[1], reverse=True))}")

    def _classes_match(self, class1: str, class2: str) -> bool:
        """Determine if two class names represent the same class"""
        if not class1 or not class2:
            return False
        
        class1_lower = class1.lower()
        class2_lower = class2.lower()
        
        # Exact match
        if class1_lower == class2_lower:
            return True
        
        # Remove common suffixes/prefixes
        suffixes = ['test', 'tests', 'it', 'spec', 'specs', 'impl', 'default']
        prefixes = ['test', 'default', 'base', 'abstract']
        
        def normalize_class_name(name):
            name = name.lower()
            for suffix in suffixes:
                if name.endswith(suffix):
                    name = name[:-len(suffix)]
            for prefix in prefixes:
                if name.startswith(prefix):
                    name = name[len(prefix):]
            return name
        
        normalized1 = normalize_class_name(class1)
        normalized2 = normalize_class_name(class2)
        
        if normalized1 == normalized2:
            return True
        
        # Check if one is contained in the other
        if normalized1 in normalized2 or normalized2 in normalized1:
            return True
        
        return False

    def save_detailed_results(self, repo_functions: Dict[str, List[ConcurrentFunction]]):
        """Save detailed analysis results with test information"""
        print(f"\n📊 Saving detailed results with test information...")
        
        # Save repository-wise analysis
        for repo_name, functions in repo_functions.items():
            repo_safe_name = repo_name.replace('/', '_')
            
            # Calculate test statistics
            functions_with_tests = [f for f in functions if f.test_functions]
            total_test_functions = sum(len(f.test_functions) for f in functions)
            
            # Organize functions by domain and complexity
            repo_data = {
                'repository': repo_name,
                'analysis_date': '2025-06-09',
                'total_functions': len(functions),
                'functions_with_tests': len(functions_with_tests),
                'test_coverage_percentage': (len(functions_with_tests) / len(functions) * 100) if functions else 0,
                'total_test_functions': total_test_functions,
                'functions_by_domain': self._group_by_domain(functions),
                'functions_by_complexity': self._group_by_complexity(functions),
                'functions_by_type': self._group_by_method_type(functions),
                'test_type_distribution': self._get_test_type_distribution(functions),
                'top_sync_primitives': self._get_top_sync_primitives(functions),
                'concurrency_patterns': self._get_pattern_distribution(functions),
                'functions': [asdict(func) for func in functions]
            }
            
            repo_file = self.output_dir / "repositories" / f"{repo_safe_name}_analysis.json"
            with open(repo_file, 'w', encoding='utf-8') as f:
                json.dump(repo_data, f, indent=2, ensure_ascii=False)
        
        # Save comprehensive summary
        all_functions = []
        for functions in repo_functions.values():
            all_functions.extend(functions)
        
        functions_with_tests = [f for f in all_functions if f.test_functions]
        total_test_functions = sum(len(f.test_functions) for f in all_functions)
        
        summary = {
            'analysis_date': '2025-06-09',
            'total_repositories': len(repo_functions),
            'total_functions': len(all_functions),
            'functions_with_tests': len(functions_with_tests),
            'test_coverage_percentage': (len(functions_with_tests) / len(all_functions) * 100) if all_functions else 0,
            'total_test_functions': total_test_functions,
            'average_functions_per_repo': len(all_functions) / len(repo_functions) if repo_functions else 0,
            'average_tests_per_function': total_test_functions / len(all_functions) if all_functions else 0,
            'domain_distribution': self._group_by_domain(all_functions),
            'complexity_distribution': self._group_by_complexity(all_functions),
            'method_type_distribution': self._group_by_method_type(all_functions),
            'test_type_distribution': self._get_test_type_distribution(all_functions),
            'top_sync_primitives': self._get_top_sync_primitives(all_functions),
            'concurrency_patterns': self._get_pattern_distribution(all_functions),
            'repositories': {
                name: {
                    'total_functions': len(funcs),
                    'functions_with_tests': len([f for f in funcs if f.test_functions]),
                    'test_coverage': (len([f for f in funcs if f.test_functions]) / len(funcs) * 100) if funcs else 0
                }
                for name, funcs in repo_functions.items()
            }
        }
        
        summary_file = self.output_dir / "analysis_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Save test-specific analysis
        test_analysis = {
            'analysis_date': '2025-06-09',
            'test_statistics': {
                'total_test_functions': total_test_functions,
                'functions_with_tests': len(functions_with_tests),
                'test_coverage_percentage': (len(functions_with_tests) / len(all_functions) * 100) if all_functions else 0,
            },
            'test_type_distribution': self._get_test_type_distribution(all_functions),
            'test_quality_metrics': self._calculate_test_quality_metrics(all_functions),
            'detailed_test_functions': [
                {
                    'function_name': func.name,
                    'class_name': func.class_name,
                    'repository': func.repository,
                    'test_count': len(func.test_functions),
                    'test_types': list(set(tf.test_type for tf in func.test_functions)),
                    'test_files': func.test_files,
                    'tests': [asdict(tf) for tf in func.test_functions]
                }
                for func in functions_with_tests
            ]
        }
        
        test_file = self.output_dir / "test_analysis.json"
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_analysis, f, indent=2, ensure_ascii=False)
        
        print(f"  📁 Repository analyses: {len(repo_functions)} files")
        print(f"  📋 Summary: analysis_summary.json")
        print(f"  🧪 Test analysis: test_analysis.json")
        print(f"  📂 Output directory: {self.output_dir}")

    def _get_test_type_distribution(self, functions: List[ConcurrentFunction]) -> Dict[str, int]:
        """Get distribution of test types"""
        test_type_counts = defaultdict(int)
        for func in functions:
            for test_func in func.test_functions:
                test_type_counts[test_func.test_type] += 1
        return dict(sorted(test_type_counts.items(), key=lambda x: x[1], reverse=True))

    def _calculate_test_quality_metrics(self, functions: List[ConcurrentFunction]) -> Dict[str, float]:
        """Calculate test quality metrics"""
        if not functions:
            return {}
        
        functions_with_tests = [f for f in functions if f.test_functions]
        
        # Test coverage
        test_coverage = len(functions_with_tests) / len(functions) * 100
        
        # Average tests per function
        avg_tests_per_function = sum(len(f.test_functions) for f in functions) / len(functions)
        
        # Test diversity (different test types per function)
        test_diversity_scores = []
        for func in functions_with_tests:
            unique_test_types = len(set(tf.test_type for tf in func.test_functions))
            test_diversity_scores.append(unique_test_types)
        
        avg_test_diversity = sum(test_diversity_scores) / len(test_diversity_scores) if test_diversity_scores else 0
        
        # Concurrency test ratio
        concurrency_test_count = 0
        total_test_count = 0
        for func in functions:
            for test_func in func.test_functions:
                total_test_count += 1
                if test_func.test_type == 'concurrency':
                    concurrency_test_count += 1
        
        concurrency_test_ratio = (concurrency_test_count / total_test_count * 100) if total_test_count > 0 else 0
        
        return {
            'test_coverage_percentage': round(test_coverage, 2),
            'average_tests_per_function': round(avg_tests_per_function, 2),
            'average_test_diversity': round(avg_test_diversity, 2),
            'concurrency_test_ratio_percentage': round(concurrency_test_ratio, 2)
        }

    def _group_by_domain(self, functions: List[ConcurrentFunction]) -> Dict[str, int]:
        """Group functions by domain"""
        domain_counts = defaultdict(int)
        for func in functions:
            domain_counts[func.domain] += 1
        return dict(sorted(domain_counts.items(), key=lambda x: x[1], reverse=True))
    
    def _group_by_complexity(self, functions: List[ConcurrentFunction]) -> Dict[str, int]:
        """Group functions by complexity level"""
        complexity_groups = {'low': 0, 'medium': 0, 'high': 0, 'very_high': 0}
        
        for func in functions:
            if func.complexity_score < 10:
                complexity_groups['low'] += 1
            elif func.complexity_score < 25:
                complexity_groups['medium'] += 1
            elif func.complexity_score < 50:
                complexity_groups['high'] += 1
            else:
                complexity_groups['very_high'] += 1
        
        return complexity_groups

    def _group_by_method_type(self, functions: List[ConcurrentFunction]) -> Dict[str, int]:
        """Group functions by method type"""
        type_counts = defaultdict(int)
        for func in functions:
            type_counts[func.method_type] += 1
        return dict(type_counts)

    def _get_top_sync_primitives(self, functions: List[ConcurrentFunction]) -> Dict[str, int]:
        """Get top synchronization primitives usage"""
        primitive_counts = defaultdict(int)
        for func in functions:
            for primitive in func.sync_primitives:
                primitive_counts[primitive] += 1
        
        # Return top 10
        return dict(sorted(primitive_counts.items(), key=lambda x: x[1], reverse=True)[:10])

    def _get_pattern_distribution(self, functions: List[ConcurrentFunction]) -> Dict[str, int]:
        """Get concurrency pattern distribution"""
        pattern_counts = defaultdict(int)
        for func in functions:
            for pattern in func.concurrency_patterns:
                pattern_counts[pattern] += 1
        
        return dict(sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True))

    def print_analysis_summary(self, repo_functions: Dict[str, List[ConcurrentFunction]]):
        """Print comprehensive analysis summary"""
        all_functions = []
        for functions in repo_functions.values():
            all_functions.extend(functions)
        
        if not all_functions:
            print("❌ No concurrent functions found!")
            return
        
        print(f"\n📊 Comprehensive Analysis Summary")
        print("=" * 50)
        
        # Basic statistics
        total_repos = len(repo_functions)
        total_functions = len(all_functions)
        functions_with_tests = len([f for f in all_functions if f.test_functions])
        total_test_functions = sum(len(f.test_functions) for f in all_functions)
        
        print(f"📈 Overall Statistics:")
        print(f"  • Repositories analyzed: {total_repos}")
        print(f"  • Concurrent functions found: {total_functions}")
        print(f"  • Functions with tests: {functions_with_tests}")
        print(f"  • Test coverage: {(functions_with_tests/total_functions*100):.1f}%")
        print(f"  • Total test functions: {total_test_functions}")
        
        # Domain distribution
        domain_dist = self._group_by_domain(all_functions)
        print(f"\n🏗️ Domain Distribution:")
        for domain, count in list(domain_dist.items())[:5]:
            print(f"  • {domain}: {count} functions")
        
        # Complexity distribution
        complexity_dist = self._group_by_complexity(all_functions)
        print(f"\n⚡ Complexity Distribution:")
        for level, count in complexity_dist.items():
            print(f"  • {level}: {count} functions")
        
        # Test type distribution
        test_type_dist = self._get_test_type_distribution(all_functions)
        if test_type_dist:
            print(f"\n🧪 Test Type Distribution:")
            for test_type, count in list(test_type_dist.items())[:5]:
                print(f"  • {test_type}: {count} tests")
        
        # Top repositories by function count
        repo_stats = [(repo, len(funcs)) for repo, funcs in repo_functions.items()]
        repo_stats.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n📂 Top Repositories:")
        for repo, count in repo_stats[:5]:
            funcs_with_tests = len([f for f in repo_functions[repo] if f.test_functions])
            coverage = (funcs_with_tests / count * 100) if count > 0 else 0
            print(f"  • {repo}: {count} functions ({coverage:.1f}% tested)")
        
        # Concurrency patterns
        pattern_dist = self._get_pattern_distribution(all_functions)
        if pattern_dist:
            print(f"\n🔄 Top Concurrency Patterns:")
            for pattern, count in list(pattern_dist.items())[:5]:
                print(f"  • {pattern}: {count} occurrences")
        
        # Synchronization primitives
        sync_dist = self._get_top_sync_primitives(all_functions)
        if sync_dist:
            print(f"\n🔒 Top Synchronization Primitives:")
            for primitive, count in list(sync_dist.items())[:5]:
                print(f"  • {primitive}: {count} usages")

def main():
    """Main analysis pipeline for refined Phase 2"""
    print("🔍 Enhanced Phase 2: Refined Java Concurrent Code Analysis")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = RefinedJavaConcurrentAnalyzer()
    
    # Phase 2.1: Clone Java repositories
    print("\n📥 Phase 2.1: Cloning Java repositories...")
    cloned_repos = analyzer.clone_selected_repositories(max_repos=10)
    
    if not cloned_repos:
        print("❌ No repositories cloned. Exiting...")
        return
    
    print(f"✅ Successfully cloned {len(cloned_repos)} repositories")
    
    # Phase 2.2: Analyze all Java files for concurrent functions
    print("\n🔍 Phase 2.2: Analyzing Java files for concurrent functions...")
    repo_functions = analyzer.analyze_java_files(cloned_repos)
    
    # Phase 2.3: Save detailed results
    print("\n💾 Phase 2.3: Saving analysis results...")
    analyzer.save_detailed_results(repo_functions)
    
    # Phase 2.4: Display comprehensive summary
    analyzer.print_analysis_summary(repo_functions)
    
    print(f"\n🎉 Analysis Complete!")
    print(f"📁 Results saved to: {analyzer.output_dir}")
    print(f"📊 Check 'analysis_summary.json' for detailed statistics")
    print(f"🧪 Check 'test_analysis.json' for test-specific insights")
    
    return repo_functions

if __name__ == "__main__":
    try:
        results = main()
        if results:
            print(f"\n✨ Successfully analyzed {sum(len(funcs) for funcs in results.values())} concurrent functions!")
        else:
            print("\n⚠️ No analysis results generated.")
    except KeyboardInterrupt:
        print("\n\n⏹️ Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()






































# import json
# import os
# import re
# import ast
# import subprocess
# from typing import List, Dict, Set, Tuple, Optional
# from dataclasses import dataclass, asdict
# from pathlib import Path
# import shutil
# from collections import defaultdict

# @dataclass
# class ConcurrentFunction:
#     """Represents a concurrent function with its properties"""
#     name: str
#     signature: str
#     file_path: str
#     language: str
#     sync_primitives: List[str]
#     dependencies: List[str]
#     concurrency_patterns: List[str]
#     complexity_score: int
#     source_code: str
#     line_number: int
#     repository: str
#     test_files: List[str] = None
#     domain: str = ""
#     class_name: str = ""
#     method_type: str = ""  # constructor, method, static_method
    
#     def __post_init__(self):
#         if self.test_files is None:
#             self.test_files = []

# class RefinedJavaConcurrentAnalyzer:
#     def __init__(self, dataset_file: str = "concurrent_structures_dataset.json", 
#                  clone_dir: str = "cloned_repositories"):
#         """Initialize analyzer with collected dataset"""
#         with open(dataset_file, 'r') as f:
#             self.dataset = json.load(f)
        
#         self.clone_dir = Path(clone_dir)
#         self.clone_dir.mkdir(exist_ok=True)
        
#         # Create organized output structure
#         self.output_dir = Path("concurrent_analysis_output")
#         self.output_dir.mkdir(exist_ok=True)
#         (self.output_dir / "repositories").mkdir(exist_ok=True)
#         (self.output_dir / "functions").mkdir(exist_ok=True)
#         (self.output_dir / "tests").mkdir(exist_ok=True)
        
#         # Enhanced Java concurrent patterns
#         self.java_concurrent_imports = [
#             'java.util.concurrent', 'java.util.concurrent.atomic', 'java.util.concurrent.locks',
#             'java.lang.Thread', 'java.util.concurrent.ExecutorService', 'java.util.concurrent.Future',
#             'java.util.concurrent.CompletableFuture', 'java.util.concurrent.ConcurrentHashMap',
#             'java.util.concurrent.ConcurrentLinkedQueue', 'java.util.concurrent.BlockingQueue',
#             'java.util.concurrent.Semaphore', 'java.util.concurrent.CountDownLatch',
#             'java.util.concurrent.CyclicBarrier', 'java.util.concurrent.Phaser'
#         ]
        
#         self.java_concurrent_keywords = [
#             # Synchronization
#             'synchronized', 'volatile', 'wait', 'notify', 'notifyAll',
#             # Atomic operations
#             'AtomicInteger', 'AtomicLong', 'AtomicReference', 'AtomicBoolean',
#             'AtomicIntegerArray', 'AtomicLongArray', 'AtomicReferenceArray',
#             'compareAndSet', 'getAndSet', 'getAndIncrement', 'getAndDecrement',
#             'getAndAdd', 'incrementAndGet', 'decrementAndGet', 'addAndGet',
#             'compareAndExchange', 'weakCompareAndSet',
#             # Locks
#             'ReentrantLock', 'ReadWriteLock', 'StampedLock', 'Lock', 'Condition',
#             'lock', 'unlock', 'tryLock', 'lockInterruptibly', 'newCondition',
#             'await', 'signal', 'signalAll',
#             # Concurrent collections
#             'ConcurrentHashMap', 'ConcurrentLinkedQueue', 'ConcurrentLinkedDeque',
#             'ConcurrentSkipListMap', 'ConcurrentSkipListSet', 'CopyOnWriteArrayList',
#             'CopyOnWriteArraySet', 'BlockingQueue', 'ArrayBlockingQueue',
#             'LinkedBlockingQueue', 'PriorityBlockingQueue', 'DelayQueue',
#             'SynchronousQueue', 'LinkedTransferQueue', 'BlockingDeque',
#             # Executors and threading
#             'ExecutorService', 'ThreadPoolExecutor', 'ScheduledExecutorService',
#             'ForkJoinPool', 'CompletableFuture', 'Future', 'Callable',
#             'Runnable', 'Thread', 'ThreadLocal', 'InheritableThreadLocal',
#             # Synchronization utilities
#             'Semaphore', 'CountDownLatch', 'CyclicBarrier', 'Phaser', 'Exchanger',
#             # Memory model
#             'final', 'static', 'ThreadSafe', 'GuardedBy', 'Immutable'
#         ]
        
#         # Domain classification keywords
#         self.domain_keywords = {
#             'data_structures': ['queue', 'stack', 'list', 'map', 'hash', 'tree', 'graph', 
#                                'collection', 'set', 'deque', 'array', 'buffer'],
#             'synchronization': ['lock', 'mutex', 'semaphore', 'barrier', 'condition', 
#                                'latch', 'synchronized', 'monitor', 'critical'],
#             'memory_management': ['allocate', 'deallocate', 'pool', 'cache', 'buffer', 
#                                 'memory', 'heap', 'reference', 'weak', 'soft'],
#             'thread_management': ['thread', 'worker', 'executor', 'scheduler', 'pool',
#                                 'fork', 'join', 'parallel', 'async', 'future'],
#             'atomic_operations': ['atomic', 'cas', 'compare', 'swap', 'increment', 
#                                 'decrement', 'exchange', 'volatile'],
#             'producer_consumer': ['producer', 'consumer', 'put', 'take', 'offer', 
#                                 'poll', 'blocking', 'queue'],
#             'utilities': ['util', 'helper', 'factory', 'builder', 'adapter', 'wrapper']
#         }

#     def clone_selected_repositories(self, max_repos: int = 10) -> List[Dict]:
#         """Clone top Java repositories for analysis"""
#         print(f"\n📥 Cloning top {max_repos} Java repositories...")
        
#         repositories = self.dataset['repositories']
#         print(f"📊 Found {len(repositories)} repositories in dataset")
        
#         # Filter Java repositories and sort by concurrent file count and stars
#         java_repos = [repo for repo in repositories if repo.get('language') == 'Java']
#         top_repos = sorted(java_repos, 
#                           key=lambda x: (
#                               x.get('structure_analysis', {}).get('concurrent_file_count', 0),
#                               x.get('stargazers_count', 0)
#                           ), 
#                           reverse=True)[:max_repos]
        
#         cloned_repos = []
#         for repo in top_repos:
#             repo_name = repo['name']
            
#             # Handle both string and dict formats for owner
#             if isinstance(repo['owner'], dict):
#                 owner = repo['owner']['login']
#             else:
#                 owner = repo['owner']
                
#             clone_url = repo['clone_url']
#             repo_path = self.clone_dir / f"{owner}_{repo_name}"
            
#             if repo_path.exists():
#                 print(f"  ✓ Repository {owner}/{repo_name} already exists")
#             else:
#                 try:
#                     print(f"  📦 Cloning {owner}/{repo_name}...")
#                     subprocess.run(['git', 'clone', '--depth', '1', clone_url, str(repo_path)], 
#                                  check=True, capture_output=True, text=True)
#                     print(f"  ✓ Successfully cloned {owner}/{repo_name}")
#                 except subprocess.CalledProcessError as e:
#                     print(f"  ❌ Failed to clone {owner}/{repo_name}: {e}")
#                     continue
            
#             repo['local_path'] = str(repo_path)
#             cloned_repos.append(repo)
        
#         print(f"📊 Successfully cloned {len(cloned_repos)} repositories")
#         return cloned_repos

#     def analyze_java_files(self, cloned_repos: List[Dict]) -> Dict[str, List[ConcurrentFunction]]:
#         """Analyze all Java files in repositories"""
#         print("\n🔍 Analyzing Java concurrent functions...")
        
#         repo_functions = {}
        
#         for repo in cloned_repos:
#             repo_name = f"{repo['owner'] if isinstance(repo['owner'], str) else repo['owner']['login']}/{repo['name']}"
#             print(f"\n  📂 Analyzing {repo_name}...")
            
#             repo_path = Path(repo['local_path'])
#             functions = []
            
#             # Find all Java files
#             java_files = list(repo_path.rglob("*.java"))
#             print(f"    📄 Found {len(java_files)} Java files")
            
#             concurrent_files = 0
#             for java_file in java_files:
#                 # Skip test files for now (analyze separately)
#                 if self._is_test_file(java_file):
#                     continue
                
#                 # Skip very large files
#                 try:
#                     file_size = java_file.stat().st_size
#                     if file_size > 500000:  # 500KB limit
#                         continue
                        
#                     with open(java_file, 'r', encoding='utf-8', errors='ignore') as f:
#                         content = f.read()
                    
#                     # Quick check if file contains concurrent code
#                     if not self._has_concurrent_code(content):
#                         continue
                    
#                     concurrent_files += 1
#                     file_functions = self.extract_java_concurrent_functions(
#                         content, str(java_file), repo_name
#                     )
#                     functions.extend(file_functions)
                    
#                 except Exception as e:
#                     print(f"    ⚠️  Error analyzing {java_file.name}: {e}")
#                     continue
            
#             print(f"    ✓ Analyzed {concurrent_files} concurrent files")
#             print(f"    ✓ Found {len(functions)} concurrent functions")
            
#             repo_functions[repo_name] = functions
        
#         return repo_functions

#     def extract_java_concurrent_functions(self, content: str, file_path: str, repo_name: str) -> List[ConcurrentFunction]:
#         """Extract concurrent functions from Java file content"""
#         functions = []
#         lines = content.split('\n')
        
#         # Get class name
#         class_name = self._extract_class_name(content)
        
#         # Find all method definitions with improved patterns
#         method_patterns = [
#             # Standard method pattern
#             r'((?:public|private|protected|static|final|abstract|synchronized|native|\s)+)\s+(\w+(?:<[^>]*>)?(?:\[\])*)\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+[^{]+)?\s*\{',
#             # Constructor pattern  
#             r'((?:public|private|protected|\s)+)\s+(' + re.escape(class_name) + r')\s*\([^)]*\)\s*(?:throws\s+[^{]+)?\s*\{',
#             # Method with generic return type
#             r'((?:public|private|protected|static|final|synchronized|\s)+)<([^>]+)>\s+(\w+(?:\[\])*)\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+[^{]+)?\s*\{'
#         ]
        
#         for i, line in enumerate(lines):
#             for pattern_idx, pattern in enumerate(method_patterns):
#                 match = re.search(pattern, line)
#                 if match:
#                     # Extract method details based on pattern
#                     if pattern_idx == 0:  # Standard method
#                         modifiers = match.group(1).strip()
#                         return_type = match.group(2)
#                         method_name = match.group(3)
#                         method_type = "constructor" if method_name == class_name else "method"
#                     elif pattern_idx == 1:  # Constructor
#                         modifiers = match.group(1).strip()
#                         method_name = match.group(2)
#                         return_type = "void"
#                         method_type = "constructor"
#                     else:  # Generic method
#                         modifiers = match.group(1).strip()
#                         generic_type = match.group(2)
#                         return_type = match.group(3)
#                         method_name = match.group(4)
#                         method_type = "method"
                    
#                     # Skip common non-methods
#                     if method_name.lower() in ['if', 'while', 'for', 'switch', 'try', 'catch', 'else']:
#                         continue
                    
#                     # Extract method body
#                     method_body = self._extract_method_body(lines, i)
#                     full_method = line + '\n' + method_body
                    
#                     # Check for concurrent elements
#                     sync_primitives = self._find_concurrent_patterns(full_method)
#                     concurrency_patterns = self._detect_high_level_patterns(full_method)
                    
#                     # Only include methods with concurrent elements
#                     if sync_primitives or concurrency_patterns or 'synchronized' in modifiers:
#                         func = ConcurrentFunction(
#                             name=method_name,
#                             signature=line.strip(),
#                             file_path=file_path,
#                             language='java',
#                             sync_primitives=sync_primitives,
#                             dependencies=self._extract_java_dependencies(full_method),
#                             concurrency_patterns=concurrency_patterns,
#                             complexity_score=self._calculate_complexity_score(full_method),
#                             source_code=method_body[:2000],  # Limit source code length
#                             line_number=i + 1,
#                             repository=repo_name,
#                             domain=self._classify_function_domain(method_name, full_method),
#                             class_name=class_name,
#                             method_type=method_type
#                         )
#                         functions.append(func)
#                     break
        
#         return functions

#     def _extract_class_name(self, content: str) -> str:
#         """Extract class name from Java file"""
#         class_match = re.search(r'(?:public\s+)?(?:abstract\s+)?(?:final\s+)?class\s+(\w+)', content)
#         if class_match:
#             return class_match.group(1)
        
#         interface_match = re.search(r'(?:public\s+)?interface\s+(\w+)', content)
#         if interface_match:
#             return interface_match.group(1)
        
#         enum_match = re.search(r'(?:public\s+)?enum\s+(\w+)', content)
#         if enum_match:
#             return enum_match.group(1)
        
#         return "Unknown"

#     def _has_concurrent_code(self, content: str) -> bool:
#         """Check if Java file contains concurrent code"""
#         content_lower = content.lower()
        
#         # Check imports
#         for import_pattern in self.java_concurrent_imports:
#             if import_pattern.lower() in content_lower:
#                 return True
        
#         # Check keywords
#         for keyword in self.java_concurrent_keywords:
#             if keyword.lower() in content_lower:
#                 return True
        
#         return False

#     def _is_test_file(self, file_path: Path) -> bool:
#         """Check if file is a test file"""
#         path_str = str(file_path).lower()
#         return any(test_indicator in path_str for test_indicator in 
#                   ['test', 'spec', 'junit', 'mock', 'stub'])

#     def _find_concurrent_patterns(self, code: str) -> List[str]:
#         """Find concurrent patterns in Java code"""
#         found_patterns = []
#         code_lower = code.lower()
        
#         for keyword in self.java_concurrent_keywords:
#             if keyword.lower() in code_lower:
#                 found_patterns.append(keyword)
        
#         # Check for method calls that indicate concurrency
#         concurrent_method_patterns = [
#             r'\.compareAndSet\s*\(',
#             r'\.getAndSet\s*\(',
#             r'\.lock\s*\(',
#             r'\.unlock\s*\(',
#             r'\.await\s*\(',
#             r'\.signal\s*\(',
#             r'\.wait\s*\(',
#             r'\.notify\s*\(',
#             r'\.submit\s*\(',
#             r'\.execute\s*\(',
#             r'\.acquire\s*\(',
#             r'\.release\s*\('
#         ]
        
#         for pattern in concurrent_method_patterns:
#             if re.search(pattern, code, re.IGNORECASE):
#                 method_name = pattern.replace(r'\.', '').replace(r'\s*\(', '')
#                 found_patterns.append(method_name)
        
#         return list(set(found_patterns))

#     def _detect_high_level_patterns(self, code: str) -> List[str]:
#         """Detect high-level concurrency patterns"""
#         patterns = []
#         code_lower = code.lower()
        
#         pattern_rules = {
#             'double_checked_locking': ['if', 'null', 'synchronized', 'volatile'],
#             'compare_and_swap': ['compareandset', 'atomic'],
#             'producer_consumer': ['blockingqueue', 'put', 'take'],
#             'reader_writer_lock': ['readwritelock', 'readlock', 'writelock'],
#             'thread_pool': ['executorservice', 'threadpool', 'submit'],
#             'future_pattern': ['future', 'completablefuture', 'get'],
#             'observer_pattern': ['notify', 'observer', 'listener'],
#             'singleton_pattern': ['synchronized', 'instance', 'static'],
#             'barrier_synchronization': ['cyclicbarrier', 'countdownlatch', 'await'],
#             'lock_free_algorithm': ['atomic', 'cas', 'compareandset', 'volatile'],
#             'immutable_object': ['final', 'immutable', 'unmodifiable'],
#             'thread_local_storage': ['threadlocal', 'get', 'set']
#         }
        
#         for pattern_name, keywords in pattern_rules.items():
#             if sum(1 for keyword in keywords if keyword in code_lower) >= len(keywords) // 2:
#                 patterns.append(pattern_name)
        
#         return patterns

#     def _extract_java_dependencies(self, code: str) -> List[str]:
#         """Extract method and class dependencies from Java code"""
#         dependencies = []
        
#         # Method calls
#         method_calls = re.findall(r'\.(\w+)\s*\(', code)
#         dependencies.extend(method_calls)
        
#         # Static method calls
#         static_calls = re.findall(r'(\w+)\.(\w+)\s*\(', code)
#         dependencies.extend([f"{cls}.{method}" for cls, method in static_calls])
        
#         # Class instantiations
#         new_objects = re.findall(r'new\s+(\w+)(?:<[^>]*>)?\s*\(', code)
#         dependencies.extend(new_objects)
        
#         # Generic type usage
#         generic_types = re.findall(r'<(\w+)(?:,\s*\w+)*>', code)
#         dependencies.extend(generic_types)
        
#         return list(set(dependencies))

#     def _calculate_complexity_score(self, code: str) -> int:
#         """Calculate complexity score for Java concurrent code"""
#         score = 0
#         code_lower = code.lower()
        
#         # Concurrency complexity weights
#         complexity_weights = {
#             'synchronized': 5,
#             'volatile': 3,
#             'atomic': 4,
#             'lock': 4,
#             'condition': 4,
#             'semaphore': 4,
#             'barrier': 5,
#             'latch': 3,
#             'executor': 3,
#             'future': 2,
#             'concurrent': 2,
#             'thread': 2
#         }
        
#         for keyword, weight in complexity_weights.items():
#             score += code_lower.count(keyword) * weight
        
#         # Control flow complexity
#         score += code_lower.count('if') * 1
#         score += code_lower.count('while') * 2
#         score += code_lower.count('for') * 1
#         score += code_lower.count('try') * 2
#         score += code_lower.count('catch') * 1
#         score += code_lower.count('finally') * 1
        
#         # Nested complexity
#         score += code.count('{') * 0.5  # Nesting penalty
        
#         return int(score)

#     def _classify_function_domain(self, func_name: str, code: str) -> str:
#         """Classify function domain based on name and code content"""
#         func_name_lower = func_name.lower()
#         code_lower = code.lower()
        
#         # Score each domain
#         domain_scores = {}
#         for domain, keywords in self.domain_keywords.items():
#             score = 0
#             for keyword in keywords:
#                 if keyword in func_name_lower:
#                     score += 3
#                 if keyword in code_lower:
#                     score += 1
#             domain_scores[domain] = score
        
#         # Return domain with highest score
#         if domain_scores:
#             best_domain = max(domain_scores, key=domain_scores.get)
#             if domain_scores[best_domain] > 0:
#                 return best_domain
        
#         return "general"

#     def _extract_method_body(self, lines: List[str], start_line: int, max_lines: int = 200) -> str:
#         """Extract method body with proper brace matching"""
#         body_lines = []
#         brace_count = 0
#         started = False
        
#         for i in range(start_line, min(len(lines), start_line + max_lines)):
#             line = lines[i]
            
#             # Skip the method signature line for the body
#             if i > start_line:
#                 body_lines.append(line)
            
#             # Count braces
#             for char in line:
#                 if char == '{':
#                     brace_count += 1
#                     started = True
#                 elif char == '}':
#                     brace_count -= 1
#                     if started and brace_count == 0:
#                         return '\n'.join(body_lines)
        
#         return '\n'.join(body_lines)

#     def save_detailed_results(self, repo_functions: Dict[str, List[ConcurrentFunction]]):
#         """Save detailed analysis results"""
#         print(f"\n📊 Saving detailed results...")
        
#         # Save repository-wise analysis
#         for repo_name, functions in repo_functions.items():
#             repo_safe_name = repo_name.replace('/', '_')
            
#             # Organize functions by domain and complexity
#             repo_data = {
#                 'repository': repo_name,
#                 'analysis_date': '2025-06-03',
#                 'total_functions': len(functions),
#                 'functions_by_domain': self._group_by_domain(functions),
#                 'functions_by_complexity': self._group_by_complexity(functions),
#                 'functions_by_type': self._group_by_method_type(functions),
#                 'top_sync_primitives': self._get_top_sync_primitives(functions),
#                 'concurrency_patterns': self._get_pattern_distribution(functions),
#                 'functions': [asdict(func) for func in functions]
#             }
            
#             repo_file = self.output_dir / "repositories" / f"{repo_safe_name}_analysis.json"
#             with open(repo_file, 'w', encoding='utf-8') as f:
#                 json.dump(repo_data, f, indent=2, ensure_ascii=False)
        
#         # Save comprehensive summary
#         all_functions = []
#         for functions in repo_functions.values():
#             all_functions.extend(functions)
        
#         summary = {
#             'analysis_date': '2025-06-03',
#             'total_repositories': len(repo_functions),
#             'total_functions': len(all_functions),
#             'average_functions_per_repo': len(all_functions) / len(repo_functions) if repo_functions else 0,
#             'domain_distribution': self._group_by_domain(all_functions),
#             'complexity_distribution': self._group_by_complexity(all_functions),
#             'method_type_distribution': self._group_by_method_type(all_functions),
#             'top_sync_primitives': self._get_top_sync_primitives(all_functions),
#             'concurrency_patterns': self._get_pattern_distribution(all_functions),
#             'repositories': {
#                 name: len(funcs) for name, funcs in repo_functions.items()
#             }
#         }
        
#         summary_file = self.output_dir / "analysis_summary.json"
#         with open(summary_file, 'w', encoding='utf-8') as f:
#             json.dump(summary, f, indent=2, ensure_ascii=False)
        
#         print(f"  📁 Repository analyses: {len(repo_functions)} files")
#         print(f"  📋 Summary: analysis_summary.json")
#         print(f"  📂 Output directory: {self.output_dir}")

#     def _group_by_domain(self, functions: List[ConcurrentFunction]) -> Dict[str, int]:
#         """Group functions by domain"""
#         domain_counts = defaultdict(int)
#         for func in functions:
#             domain_counts[func.domain] += 1
#         return dict(sorted(domain_counts.items(), key=lambda x: x[1], reverse=True))

#     def _group_by_complexity(self, functions: List[ConcurrentFunction]) -> Dict[str, int]:
#         """Group functions by complexity level"""
#         complexity_groups = {'low': 0, 'medium': 0, 'high': 0, 'very_high': 0}
        
#         for func in functions:
#             if func.complexity_score < 10:
#                 complexity_groups['low'] += 1
#             elif func.complexity_score < 25:
#                 complexity_groups['medium'] += 1
#             elif func.complexity_score < 50:
#                 complexity_groups['high'] += 1
#             else:
#                 complexity_groups['very_high'] += 1
        
#         return complexity_groups

#     def _group_by_method_type(self, functions: List[ConcurrentFunction]) -> Dict[str, int]:
#         """Group functions by method type"""
#         type_counts = defaultdict(int)
#         for func in functions:
#             type_counts[func.method_type] += 1
#         return dict(type_counts)

#     def _get_top_sync_primitives(self, functions: List[ConcurrentFunction]) -> Dict[str, int]:
#         """Get top synchronization primitives usage"""
#         primitive_counts = defaultdict(int)
#         for func in functions:
#             for primitive in func.sync_primitives:
#                 primitive_counts[primitive] += 1
        
#         # Return top 10
#         return dict(sorted(primitive_counts.items(), key=lambda x: x[1], reverse=True)[:10])

#     def _get_pattern_distribution(self, functions: List[ConcurrentFunction]) -> Dict[str, int]:
#         """Get concurrency pattern distribution"""
#         pattern_counts = defaultdict(int)
#         for func in functions:
#             for pattern in func.concurrency_patterns:
#                 pattern_counts[pattern] += 1
        
#         return dict(sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True))

# def main():
#     """Main analysis pipeline for refined Phase 2"""
#     print("🔍 Enhanced Phase 2: Refined Java Concurrent Code Analysis")
#     print("=" * 70)
    
#     # Initialize analyzer
#     analyzer = RefinedJavaConcurrentAnalyzer()
    
#     # Phase 2.1: Clone Java repositories
#     cloned_repos = analyzer.clone_selected_repositories(max_repos=10)
    
#     if not cloned_repos:
#         print("❌ No repositories cloned. Exiting...")
#         return
    
#     # Phase 2.2: Analyze all Java files for concurrent functions
#     repo_functions = analyzer.analyze_java_files(cloned_repos)
    
#     # Phase 2.3: Save detailed results
#     analyzer.save_detailed_results(repo_functions)
    
#     # Display comprehensive summary
#     total_functions = sum(len(funcs) for funcs in repo_functions.values())
    
#     print(f"\n📈 Analysis Complete!")
#     print(f"  🔍 Repositories Analyzed: {len(repo_functions)}")
#     print(f"  ⚡ Total Concurrent Functions: {total_functions}")
#     print(f"  📊 Average Functions/Repo: {total_functions/len(repo_functions):.1f}")
    
#     # Show detailed breakdown
#     print(f"\n🔧 Detailed Breakdown by Repository:")
#     for repo_name, functions in repo_functions.items():
#         if functions:
#             domains = set(func.domain for func in functions)
#             patterns = set(pattern for func in functions for pattern in func.concurrency_patterns)
#             primitives = set(prim for func in functions for prim in func.sync_primitives)
            
#             print(f"  📂 {repo_name}: {len(functions)} functions")
#             print(f"    • Domains: {', '.join(list(domains)[:3])}")
#             print(f"    • Patterns: {len(patterns)} unique patterns")
#             print(f"    • Primitives: {len(primitives)} unique primitives")
#         else:
#             print(f"  📂 {repo_name}: 0 functions")
    
#     return repo_functions

# if __name__ == "__main__":
#     results = main()