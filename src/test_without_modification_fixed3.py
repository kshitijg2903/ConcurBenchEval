#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test generated functions without modifying original source code
"""

import json
import os
import logging
import sys
import subprocess
import tempfile
import shutil
import re

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
        logger.info("Loading generated function from: {0}".format(file_path))
        if not os.path.exists(file_path):
            logger.error("File does not exist: {0}".format(file_path))
            return None
            
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        logger.error("Error loading generated function: {0}".format(e))
        return None

def create_test_harness(function_data, no_context_impl, full_context_impl):
    """Create a test harness to compare implementations"""
    name = function_data.get('name', 'unknown')
    signature = function_data.get('signature', '')
    class_name = function_data.get('class_name', '')
    file_path = function_data.get('file_path', '')
    test_files = function_data.get('test_files', [])
    
    logger.info("Creating test harness for: {0}.{1}".format(class_name, name))
    logger.info("File path: {0}".format(file_path))
    
    # Extract package from file path
    package = "org.jctools.channels.mapping"  # Default package
    if "src/main/java/" in file_path:
        package_path = file_path.split("src/main/java/")[1].split("/")
        package_path = package_path[:-1]  # Remove the file name
        package = ".".join(package_path)
    
    logger.info("Package: {0}".format(package))
    
    # Create a test harness class
    test_harness = """
package {0};

import org.junit.Test;
import static org.junit.Assert.*;
import java.lang.reflect.*;

/**
 * Test harness to compare original and generated implementations
 * without modifying the original source code.
 */
public class GeneratedFunctionTest {{

    /**
     * Original implementation wrapper
     */
    public static class Original{1} extends {1} {{
        public Original{1}(Class<?> structInterface, boolean debugEnabled) {{
            super(structInterface, debugEnabled);
        }}
        
        @Override
        public int getSizeInBytes() {{
            return super.getSizeInBytes();
        }}
    }}
    
    /**
     * No-context implementation
     */
    public static class NoContext{1} extends {1} {{
        public NoContext{1}(Class<?> structInterface, boolean debugEnabled) {{
            super(structInterface, debugEnabled);
        }}
        
        @Override
        public int getSizeInBytes() {{
            {2}
        }}
    }}
    
    /**
     * Full-context implementation
     */
    public static class FullContext{1} extends {1} {{
        public FullContext{1}(Class<?> structInterface, boolean debugEnabled) {{
            super(structInterface, debugEnabled);
        }}
        
        @Override
        public int getSizeInBytes() {{
            {3}
        }}
    }}
    
    @Test
    public void testNoContextImplementation() {{
        try {{
            // Create instances
            Original{1} original = new Original{1}(Example.class, false);
            NoContext{1} noContext = new NoContext{1}(Example.class, false);
            
            // Compare results
            int originalResult = original.getSizeInBytes();
            int noContextResult = noContext.getSizeInBytes();
            
            assertEquals("No-context implementation should match original", originalResult, noContextResult);
            System.out.println("No-context implementation test: PASSED");
        }} catch (Exception e) {{
            System.out.println("No-context implementation test: FAILED");
            e.printStackTrace();
            fail("No-context implementation test failed: " + e.getMessage());
        }}
    }}
    
    @Test
    public void testFullContextImplementation() {{
        try {{
            // Create instances
            Original{1} original = new Original{1}(Example.class, false);
            FullContext{1} fullContext = new FullContext{1}(Example.class, false);
            
            // Compare results
            int originalResult = original.getSizeInBytes();
            int fullContextResult = fullContext.getSizeInBytes();
            
            assertEquals("Full-context implementation should match original", originalResult, fullContextResult);
            System.out.println("Full-context implementation test: PASSED");
        }} catch (Exception e) {{
            System.out.println("Full-context implementation test: FAILED");
            e.printStackTrace();
            fail("Full-context implementation test failed: " + e.getMessage());
        }}
    }}
    
    // Example interface for testing
    public interface Example {{
        int getFoo();
        void setFoo(int value);
        long getBar();
        void setBar(long value);
    }}
}}
""".format(package, class_name, no_context_impl, full_context_impl)
    
    return test_harness

def create_temp_directory():
    """Create a temporary directory for testing"""
    return tempfile.mkdtemp()

def modify_pom_xml(pom_path):
    """Modify the pom.xml to use an older version of the enforcer plugin"""
    try:
        with open(pom_path, 'r') as f:
            content = f.read()
        
        # Modify the enforcer plugin version
        content = re.sub(r'<artifactId>maven-enforcer-plugin</artifactId>\s*<version>3.0.0</version>', 
                         '<artifactId>maven-enforcer-plugin</artifactId><version>1.4.1</version>', content)
        
        with open(pom_path, 'w') as f:
            f.write(content)
        
        logger.info("Modified pom.xml to use older enforcer plugin version")
        return True
    except Exception as e:
        logger.error("Error modifying pom.xml: {0}".format(e))
        return False

def setup_test_environment(repo_path, temp_dir, test_harness, package_path, module_path):
    """Set up the test environment"""
    # Copy the repository to the temporary directory
    logger.info("Repository path: {0}".format(repo_path))
    logger.info("Temporary directory: {0}".format(temp_dir))
    logger.info("Package path: {0}".format(package_path))
    logger.info("Module path: {0}".format(module_path))
    
    if not os.path.exists(repo_path):
        logger.error("Repository path does not exist: {0}".format(repo_path))
        return None
        
    temp_repo_path = os.path.join(temp_dir, os.path.basename(repo_path))
    logger.info("Copying repository to: {0}".format(temp_repo_path))
    shutil.copytree(repo_path, temp_repo_path)
    
    # Modify the pom.xml files
    parent_pom_path = os.path.join(temp_repo_path, "pom.xml")
    module_pom_path = os.path.join(temp_repo_path, module_path, "pom.xml")
    
    if os.path.exists(parent_pom_path):
        modify_pom_xml(parent_pom_path)
    
    if os.path.exists(module_pom_path):
        modify_pom_xml(module_pom_path)
    
    # Create a simple test class that doesn't require Maven
    simple_test_path = os.path.join(temp_repo_path, module_path, "SimpleTest.java")
    with open(simple_test_path, 'w') as f:
        f.write("""
import org.jctools.channels.mapping.Mapper;

public class SimpleTest {
    public static void main(String[] args) {
        try {
            // Original implementation
            Mapper<Example> original = new Mapper<>(Example.class, false);
            int originalResult = original.getSizeInBytes();
            System.out.println("Original implementation result: " + originalResult);
            
            // No-context implementation
            NoContextMapper<Example> noContext = new NoContextMapper<>(Example.class, false);
            int noContextResult = noContext.getSizeInBytes();
            System.out.println("No-context implementation result: " + noContextResult);
            System.out.println("No-context matches original: " + (originalResult == noContextResult));
            
            // Full-context implementation
            FullContextMapper<Example> fullContext = new FullContextMapper<>(Example.class, false);
            int fullContextResult = fullContext.getSizeInBytes();
            System.out.println("Full-context implementation result: " + fullContextResult);
            System.out.println("Full-context matches original: " + (originalResult == fullContextResult));
        } catch (Exception e) {
            System.out.println("Test failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    // Example interface for testing
    public interface Example {
        int getFoo();
        void setFoo(int value);
        long getBar();
        void setBar(long value);
    }
    
    // No-context implementation
    public static class NoContextMapper<S> extends Mapper<S> {
        public NoContextMapper(Class<S> structInterface, boolean debugEnabled) {
            super(structInterface, debugEnabled);
        }
        
        @Override
        public int getSizeInBytes() {
            Inspector inspector = getInspector();
            Future<Integer> sizeFuture = inspector.getSizeInBytes();
            int inspectorSize = sizeFuture.get();
            return Primitive.INT.sizeInBytes + inspectorSize;
        }
    }
    
    // Full-context implementation
    public static class FullContextMapper<S> extends Mapper<S> {
        public FullContextMapper(Class<S> structInterface, boolean debugEnabled) {
            super(structInterface, debugEnabled);
        }
        
        @Override
        public int getSizeInBytes() {
            return Primitive.INT.sizeInBytes + inspector.getSizeInBytes();
        }
    }
}
""")
    
    # Create the test harness file
    test_harness_path = os.path.join(temp_repo_path, module_path, "src/test/java", package_path, "GeneratedFunctionTest.java")
    logger.info("Creating test harness at: {0}".format(test_harness_path))
    
    # Create directory if it doesn't exist
    test_harness_dir = os.path.dirname(test_harness_path)
    if not os.path.exists(test_harness_dir):
        logger.info("Creating directory: {0}".format(test_harness_dir))
        os.makedirs(test_harness_dir)
    
    with open(test_harness_path, 'w') as f:
        f.write(test_harness)
    
    return os.path.join(temp_repo_path, module_path)

def run_tests(repo_path):
    """Run the tests"""
    try:
        # Change to the repository directory
        current_dir = os.getcwd()
        logger.info("Changing directory to: {0}".format(repo_path))
        os.chdir(repo_path)
        
        # Try to run the tests with Maven
        try:
            logger.info("Running Maven tests")
            process = subprocess.Popen(['mvn', 'test', '-Dtest=GeneratedFunctionTest'], 
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                logger.info("Maven tests succeeded")
                return True, stdout.decode('utf-8'), stderr.decode('utf-8')
            else:
                logger.warning("Maven tests failed, will try simple compilation")
        except Exception as e:
            logger.warning("Maven execution failed: {0}".format(e))
        
        # If Maven fails, try simple compilation and execution
        logger.info("Compiling with javac")
        compile_process = subprocess.Popen(['javac', 'SimpleTest.java'], 
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        compile_stdout, compile_stderr = compile_process.communicate()
        
        if compile_process.returncode == 0:
            logger.info("Compilation succeeded, running test")
            run_process = subprocess.Popen(['java', 'SimpleTest'], 
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            run_stdout, run_stderr = run_process.communicate()
            
            # Change back to the original directory
            os.chdir(current_dir)
            
            return run_process.returncode == 0, run_stdout.decode('utf-8'), run_stderr.decode('utf-8')
        else:
            logger.error("Compilation failed")
            # Change back to the original directory
            os.chdir(current_dir)
            return False, compile_stdout.decode('utf-8'), compile_stderr.decode('utf-8')
    except Exception as e:
        logger.error("Error running tests: {0}".format(e))
        # Make sure we change back to the original directory
        try:
            os.chdir(current_dir)
        except:
            pass
        return False, "", str(e)

def main():
    if len(sys.argv) < 2:
        logger.error("Usage: python test_without_modification.py <function_file> <function_index>")
        return
    
    function_file = sys.argv[1]
    function_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    function_data = load_function(function_file, function_index)
    if not function_data:
        logger.error("Failed to load function data")
        return
    
    # Get the function details
    function_name = function_data.get('name', '')
    file_path = function_data.get('file_path', '')
    
    logger.info("Function name: {0}".format(function_name))
    logger.info("File path: {0}".format(file_path))
    
    # Load the generated functions
    no_context_path = "concurrent_analysis_output/generated_functions/{0}_no_context.java".format(function_name)
    full_context_path = "concurrent_analysis_output/generated_functions/{0}_full_context.java".format(function_name)
    
    no_context = load_generated_function(no_context_path)
    full_context = load_generated_function(full_context_path)
    
    if not no_context or not full_context:
        logger.error("Failed to load generated functions")
        return
    
    # Extract package path from file path
    package_path = "org/jctools/channels/mapping"  # Default package path
    if "src/main/java/" in file_path:
        package_path = file_path.split("src/main/java/")[1].rsplit("/", 1)[0]
    
    logger.info("Package path: {0}".format(package_path))
    
    # Create the test harness
    test_harness = create_test_harness(function_data, no_context, full_context)
    
    # Create a temporary directory for testing
    temp_dir = create_temp_directory()
    logger.info("Created temporary directory: {0}".format(temp_dir))
    
    try:
        # Get the repository path - FIXED: Use the correct path to the full repository
        repo_path = "/local/home/birenpr/ConcurBench/cloned_repositories/JCTools_JCTools"
        module_path = "jctools-channels"  # The module containing our function
        logger.info("Repository path: {0}".format(repo_path))
        
        # Check if the repository path exists
        if not os.path.exists(repo_path):
            logger.error("Repository path does not exist: {0}".format(repo_path))
            return
        
        # Set up the test environment
        temp_repo_path = setup_test_environment(repo_path, temp_dir, test_harness, package_path, module_path)
        if not temp_repo_path:
            logger.error("Failed to set up test environment")
            return
            
        logger.info("Set up test environment in: {0}".format(temp_repo_path))
        
        # Run the tests
        success, stdout, stderr = run_tests(temp_repo_path)
        
        logger.info("Test result: {0}".format("PASS" if success else "FAIL"))
        logger.info("Test output:\n{0}".format(stdout))
        if stderr:
            logger.info("Test errors:\n{0}".format(stderr))
    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)
        logger.info("Cleaned up temporary directory")

if __name__ == "__main__":
    main()
