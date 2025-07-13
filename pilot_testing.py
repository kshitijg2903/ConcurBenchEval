#!/usr/bin/env python3
"""
Simple script to select the best 10 function-test pairs from test_analysis.json
"""

import json
from typing import List, Dict, Any

def calculate_pair_quality_score(test_info: Dict[str, Any]) -> float:
    """Calculate quality score for a function-test pair."""
    score = 0.0
    
    # Test count (more tests = better)
    test_count = test_info.get('test_count', 0)
    score += min(test_count * 3, 15)  # Max 15 points
    
    # Test comprehensiveness (longer tests = more comprehensive)
    if test_info.get('tests'):
        avg_test_length = sum(len(test.get('source_code', '')) for test in test_info['tests']) / len(test_info['tests'])
        score += min(avg_test_length / 50, 10)  # Max 10 points
    
    # Method diversity (tests calling multiple methods)
    unique_methods = set()
    for test in test_info.get('tests', []):
        unique_methods.update(test.get('tested_methods', []))
    score += min(len(unique_methods) * 2, 10)  # Max 10 points
    
    # Repository quality bonus
    repo = test_info.get('repository', '')
    if 'JCTools' in repo:
        score += 8  # JCTools is high quality
    elif 'disruptor' in repo.lower():
        score += 6
    else:
        score += 3
    
    return score

def select_top_10_pairs() -> List[Dict[str, Any]]:
    """Load test data and select top 10 pairs."""
    
    try:
        with open('test_analysis.json', 'r') as f:
            test_data = json.load(f)
    except FileNotFoundError:
        print("❌ Error: test_analysis.json not found!")
        return []
    
    detailed_functions = test_data.get('detailed_test_functions', [])
    print(f"📊 Found {len(detailed_functions)} function-test pairs to analyze")
    
    # Score each pair
    scored_pairs = []
    for test_info in detailed_functions:
        score = calculate_pair_quality_score(test_info)
        scored_pairs.append({
            'score': score,
            'data': test_info
        })
    
    # Sort by score (highest first)
    scored_pairs.sort(key=lambda x: x['score'], reverse=True)
    
    # Return top 10
    return scored_pairs[:10]

def print_pair_summary(pair_data: Dict[str, Any], rank: int):
    """Print summary of a function-test pair."""
    data = pair_data['data']
    score = pair_data['score']
    
    print(f"\n{'='*60}")
    print(f"RANK #{rank} - {data['function_name']}")
    print(f"{'='*60}")
    print(f"Class: {data['class_name']}")
    print(f"Repository: {data['repository']}")
    print(f"Quality Score: {score:.1f}")
    print(f"Test Count: {data['test_count']}")
    
    # Show test methods
    print(f"\nTest Methods:")
    for i, test in enumerate(data.get('tests', [])[:3]):  # Show first 3
        print(f"  {i+1}. {test.get('name', 'Unknown')}")
        print(f"     Tests methods: {', '.join(test.get('tested_methods', [])[:3])}")
    
    if len(data.get('tests', [])) > 3:
        print(f"  ... and {len(data.get('tests', [])) - 3} more tests")

def export_pilot_dataset(top_pairs: List[Dict[str, Any]]):
    """Export the selected pairs to pilot_dataset.json."""
    
    pilot_data = {
        'metadata': {
            'selection_date': '2025-06-09',
            'total_candidates': len(top_pairs),
            'selection_criteria': 'Quality score based on test count, comprehensiveness, and repository quality'
        },
        'pilot_pairs': []
    }
    
    for i, pair in enumerate(top_pairs):
        pilot_data['pilot_pairs'].append({
            'rank': i + 1,
            'quality_score': pair['score'],
            'function_name': pair['data']['function_name'],
            'class_name': pair['data']['class_name'],
            'repository': pair['data']['repository'],
            'test_count': pair['data']['test_count'],
            'full_data': pair['data']
        })
    
    with open('pilot_dataset.json', 'w') as f:
        json.dump(pilot_data, f, indent=2)
    
    print(f"\n✅ Exported top 10 pairs to pilot_dataset.json")

def main():
    print("🎯 Selecting Top 10 Function-Test Pairs for Pilot Study")
    print("="*60)
    
    # Get top 10 pairs
    top_pairs = select_top_10_pairs()
    
    if not top_pairs:
        print("❌ No pairs found!")
        return
    
    print(f"\n🏆 TOP 10 FUNCTION-TEST PAIRS")
    print("="*60)
    
    # Show each pair
    for i, pair in enumerate(top_pairs):
        print_pair_summary(pair, i + 1)
    
    # Export to file
    export_pilot_dataset(top_pairs)
    
    # Summary stats
    avg_score = sum(p['score'] for p in top_pairs) / len(top_pairs)
    avg_tests = sum(p['data']['test_count'] for p in top_pairs) / len(top_pairs)
    
    print(f"\n📈 SUMMARY")
    print(f"Average Quality Score: {avg_score:.1f}")
    print(f"Average Test Count: {avg_tests:.1f}")
    
    # Repository breakdown
    repos = {}
    for pair in top_pairs:
        repo = pair['data']['repository'].split('/')[-1]
        repos[repo] = repos.get(repo, 0) + 1
    
    print(f"\nRepository Distribution:")
    for repo, count in repos.items():
        print(f"  {repo}: {count} pairs")
    
    print(f"\n🚀 Ready for Phase 6A: Generate requirements for these 10 pairs!")

if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# """
# Select best 10 function-test pairs for LLM annotation pilot study.
# Prioritizes functions with good test coverage and clear concurrency patterns.
# """

# import json
# import os
# from typing import List, Dict, Any
# from dataclasses import dataclass
# from pathlib import Path

# @dataclass
# class PilotCandidate:
#     function_name: str
#     class_name: str
#     repository: str
#     test_count: int
#     complexity_score: int
#     concurrency_patterns: List[str]
#     synchronization_primitives: List[str]
#     test_quality_score: float
#     pilot_score: float
#     function_info: Dict[str, Any]
#     test_info: Dict[str, Any]

# def load_data():
#     """Load all the analysis data files."""
#     # First, let's check what files are available
#     available_files = []
#     for filename in ['concurrent_structures_dataset.json', 'functions_analysis.json', 'test_analysis.json']:
#         if os.path.exists(filename):
#             available_files.append(filename)
    
#     print(f"📁 Available files: {available_files}")
    
#     try:
#         # Try the actual filename first
#         functions_file = 'concurrent_structures_dataset.json'
#         if not os.path.exists(functions_file):
#             functions_file = 'functions_analysis.json'
        
#         with open(functions_file, 'r') as f:
#             functions_data = json.load(f)
        
#         with open('test_analysis.json', 'r') as f:
#             test_data = json.load(f)
        
#         print(f"✅ Loaded functions data from: {functions_file}")
#         print(f"✅ Loaded test data from: test_analysis.json")
        
#         return functions_data, test_data
#     except FileNotFoundError as e:
#         print(f"Error: Could not find required files. Make sure you're in the correct directory.")
#         print(f"Missing file: {e.filename}")
#         print(f"Available files in current directory:")
#         for f in os.listdir('.'):
#             if f.endswith('.json'):
#                 print(f"  - {f}")
#         return None, None

# def calculate_test_quality_score(test_info: Dict[str, Any]) -> float:
#     """Calculate a quality score for the test coverage."""
#     score = 0.0
    
#     # Base score from test count (diminishing returns)
#     test_count = test_info.get('test_count', 0)
#     score += min(test_count * 2, 10)  # Max 10 points for test count
    
#     # Bonus for diverse test methods
#     unique_tested_methods = set()
#     for test in test_info.get('tests', []):
#         unique_tested_methods.update(test.get('tested_methods', []))
    
#     score += min(len(unique_tested_methods) * 1.5, 8)  # Max 8 points for method diversity
    
#     # Bonus for test code length (more comprehensive tests)
#     avg_test_length = 0
#     if test_info.get('tests'):
#         total_length = sum(len(test.get('source_code', '')) for test in test_info['tests'])
#         avg_test_length = total_length / len(test_info['tests'])
    
#     score += min(avg_test_length / 100, 5)  # Max 5 points for test comprehensiveness
    
#     # Bonus for multiple test types
#     test_types = len(set(test_info.get('test_types', [])))
#     score += test_types * 2  # 2 points per test type
    
#     return score

# def calculate_pilot_score(func_info: Dict[str, Any], test_info: Dict[str, Any]) -> float:
#     """Calculate overall pilot suitability score."""
#     score = 0.0
    
#     # Test quality (40% weight)
#     test_quality = calculate_test_quality_score(test_info)
#     score += test_quality * 0.4
    
#     # Complexity score (30% weight) - prefer medium complexity
#     complexity = func_info.get('complexity_score', 0)
#     if 8 <= complexity <= 25:  # Sweet spot for pilot
#         score += 15 * 0.3
#     elif 5 <= complexity <= 35:
#         score += 10 * 0.3
#     else:
#         score += 5 * 0.3
    
#     # Concurrency richness (20% weight)
#     patterns = len(func_info.get('concurrency_patterns', []))
#     primitives = len(func_info.get('synchronization_primitives', []))
#     concurrency_richness = min(patterns + primitives, 10)
#     score += concurrency_richness * 0.2
    
#     # Repository diversity bonus (10% weight)
#     repo_name = func_info.get('repository', '')
#     if 'JCTools' in repo_name:
#         score += 8 * 0.1  # JCTools has high-quality concurrent code
#     elif 'disruptor' in repo_name.lower():
#         score += 7 * 0.1
#     else:
#         score += 5 * 0.1
    
#     return score

# def select_pilot_candidates(functions_data: Dict, test_data: Dict) -> List[PilotCandidate]:
#     """Select and rank the best pilot candidates."""
#     candidates = []
    
#     # Create lookup for test data
#     test_lookup = {}
#     for test_info in test_data.get('detailed_test_functions', []):
#         key = f"{test_info['class_name']}.{test_info['function_name']}"
#         test_lookup[key] = test_info
    
#     # Handle different data structures
#     functions_list = []
#     if 'functions' in functions_data:
#         functions_list = functions_data['functions']
#     elif 'concurrent_functions' in functions_data:
#         functions_list = functions_data['concurrent_functions']
#     elif 'repositories' in functions_data:
#         # Extract functions from repositories structure
#         print(f"📁 Found repositories structure, extracting functions...")
#         for repo_data in functions_data['repositories']:
#             if 'functions' in repo_data:
#                 functions_list.extend(repo_data['functions'])
#             elif 'concurrent_functions' in repo_data:
#                 functions_list.extend(repo_data['concurrent_functions'])
#     elif isinstance(functions_data, list):
#         functions_list = functions_data
#     else:
#         print(f"⚠️  Unknown functions data structure. Keys: {list(functions_data.keys())}")
#         # Let's inspect the structure more deeply
#         print("📋 Detailed structure inspection:")
#         if 'repositories' in functions_data:
#             repos = functions_data['repositories']
#             print(f"  Found {len(repos)} repositories")
#             if repos:
#                 sample_repo = repos[0]
#                 print(f"  Sample repository keys: {list(sample_repo.keys())}")
#         return candidates
    
#     print(f"📊 Processing {len(functions_list)} functions...")
    
#     # Process each function
#     for func_info in functions_list:
#         func_key = f"{func_info.get('class_name', '')}.{func_info.get('function_name', '')}"
        
#         if func_key in test_lookup:
#             test_info = test_lookup[func_key]
            
#             # Skip if very low test count
#             if test_info.get('test_count', 0) < 1:
#                 continue
            
#             # Calculate scores
#             test_quality = calculate_test_quality_score(test_info)
#             pilot_score = calculate_pilot_score(func_info, test_info)
            
#             candidate = PilotCandidate(
#                 function_name=func_info.get('function_name', ''),
#                 class_name=func_info.get('class_name', ''),
#                 repository=func_info.get('repository', ''),
#                 test_count=test_info.get('test_count', 0),
#                 complexity_score=func_info.get('complexity_score', 0),
#                 concurrency_patterns=func_info.get('concurrency_patterns', []),
#                 synchronization_primitives=func_info.get('synchronization_primitives', []),
#                 test_quality_score=test_quality,
#                 pilot_score=pilot_score,
#                 function_info=func_info,
#                 test_info=test_info
#             )
            
#             candidates.append(candidate)
    
#     # Sort by pilot score (descending)
#     candidates.sort(key=lambda x: x.pilot_score, reverse=True)
    
#     return candidates

# def print_candidate_summary(candidate: PilotCandidate, rank: int):
#     """Print a summary of a pilot candidate."""
#     print(f"\n{'='*60}")
#     print(f"RANK #{rank} - {candidate.function_name}")
#     print(f"{'='*60}")
#     print(f"Class: {candidate.class_name}")
#     print(f"Repository: {candidate.repository}")
#     print(f"Pilot Score: {candidate.pilot_score:.1f}")
#     print(f"Test Quality Score: {candidate.test_quality_score:.1f}")
#     print(f"Test Count: {candidate.test_count}")
#     print(f"Complexity Score: {candidate.complexity_score}")
#     print(f"Concurrency Patterns: {', '.join(candidate.concurrency_patterns) if candidate.concurrency_patterns else 'None'}")
#     print(f"Sync Primitives: {', '.join(candidate.synchronization_primitives) if candidate.synchronization_primitives else 'None'}")
    
#     # Show function signature
#     func_info = candidate.function_info
#     print(f"\nFunction Signature:")
#     print(f"  {func_info.get('signature', 'N/A')}")
    
#     # Show test methods
#     print(f"\nTest Methods ({candidate.test_count}):")
#     for i, test in enumerate(candidate.test_info.get('tests', [])[:3]):  # Show first 3 tests
#         print(f"  {i+1}. {test.get('name', 'Unknown')}")
    
#     if len(candidate.test_info.get('tests', [])) > 3:
#         print(f"  ... and {len(candidate.test_info.get('tests', [])) - 3} more")

# def export_pilot_dataset(candidates: List[PilotCandidate], output_file: str = 'pilot_dataset.json'):
#     """Export the selected pilot candidates to a JSON file."""
#     pilot_data = {
#         'selection_criteria': {
#             'total_candidates_evaluated': len(candidates),
#             'selection_methodology': 'Automated scoring based on test quality, complexity, and concurrency patterns',
#             'scoring_weights': {
#                 'test_quality': 0.4,
#                 'complexity_appropriateness': 0.3,
#                 'concurrency_richness': 0.2,
#                 'repository_quality': 0.1
#             }
#         },
#         'pilot_functions': []
#     }
    
#     for i, candidate in enumerate(candidates[:10]):
#         pilot_data['pilot_functions'].append({
#             'rank': i + 1,
#             'function_name': candidate.function_name,
#             'class_name': candidate.class_name,
#             'repository': candidate.repository,
#             'pilot_score': candidate.pilot_score,
#             'test_quality_score': candidate.test_quality_score,
#             'function_details': candidate.function_info,
#             'test_details': candidate.test_info
#         })
    
#     with open(output_file, 'w') as f:
#         json.dump(pilot_data, f, indent=2)
    
#     print(f"\n✅ Pilot dataset exported to {output_file}")

# def main():
#     print("🎯 Selecting Best 10 Function-Test Pairs for LLM Annotation Pilot")
#     print("="*70)
    
#     # Load data
#     functions_data, test_data = load_data()
#     if not functions_data or not test_data:
#         return
    
#     # Select candidates
#     print(f"📊 Analyzing {len(test_data.get('detailed_test_functions', []))} tested functions...")
#     candidates = select_pilot_candidates(functions_data, test_data)
    
#     if not candidates:
#         print("❌ No suitable candidates found!")
#         return
    
#     print(f"✅ Found {len(candidates)} candidates with tests")
#     print(f"📋 Selecting top 10 for pilot study...")
    
#     # Show top 10
#     top_10 = candidates[:10]
    
#     print(f"\n🏆 TOP 10 PILOT CANDIDATES")
#     print("="*70)
    
#     for i, candidate in enumerate(top_10):
#         print_candidate_summary(candidate, i + 1)
    
#     # Export dataset
#     export_pilot_dataset(candidates)
    
#     # Summary statistics
#     print(f"\n📈 PILOT SELECTION SUMMARY")
#     print("="*50)
#     print(f"Total functions analyzed: {len(candidates)}")
#     print(f"Selected for pilot: 10")
#     print(f"Average pilot score: {sum(c.pilot_score for c in top_10) / 10:.1f}")
#     print(f"Average test count: {sum(c.test_count for c in top_10) / 10:.1f}")
#     print(f"Average complexity: {sum(c.complexity_score for c in top_10) / 10:.1f}")
    
#     # Repository distribution
#     repo_dist = {}
#     for c in top_10:
#         repo = c.repository.split('/')[-1]  # Get repo name
#         repo_dist[repo] = repo_dist.get(repo, 0) + 1
    
#     print(f"\nRepository distribution:")
#     for repo, count in repo_dist.items():
#         print(f"  {repo}: {count} functions")
    
#     print(f"\n🚀 Ready to start Phase 6A: LLM Annotation Pipeline!")
#     print(f"📁 Next step: Review pilot_dataset.json and begin requirement generation")

# if __name__ == "__main__":
#     main()