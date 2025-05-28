import os
import logging

logger = logging.getLogger(__name__)

def extract_java_info(file_path):
    """
    Extracts the package name and class name from a Java file path,
    handling both source and test files, assuming the standard
    Maven/Gradle project structure (src/main/java or src/test/java).

    Args:
        file_path (str): The path to the Java file.

    Returns:
        tuple: A tuple containing the package name (str) and class name (str).
               Returns (None, None) if the path is invalid or doesn't
               contain the expected structure.
    """
    # Check if the file path is a string
    if not isinstance(file_path, str):
        return None, None

    # Normalize the file path to handle different path separators
    file_path = os.path.normpath(file_path)

    # Split the path into its components
    parts = file_path.split(os.sep)

    src_index = -1 # Initialize src_index
    # Find the index of "src/(main|test)/java"
    try:
        src_index = parts.index("src")
        # Check for either "main" or "test"
        main_index = parts.index("main", src_index + 1)
        java_index = parts.index("java", main_index + 1)
    except ValueError:
        try:
            src_index = parts.index("src") # add this line
            main_index = parts.index("test", src_index + 1)
            java_index = parts.index("java", main_index + 1)
        except ValueError:
            return None, None  # Return None, None if "src/(main|test)/java" is not found

    # Extract package path
    package_path = parts[java_index + 1:-1]  # Exclude "src/main/java" and the file name
    package_name = ".".join(package_path)  # Convert path to package name

    # Extract class name
    file_name = parts[-1]
    if file_name.endswith(".java"):
        class_name = file_name[:-5]  # Remove ".java" extension
    else:
        return None, None  # handles the case where the file doesn't end with .java

    return package_name, class_name


if __name__ == "__main__":
    test_cases = [
        ("beanmother-core/src/main/java/io/beanmother/core/util/PrimitiveTypeUtils.java", ("io.beanmother.core.util", "PrimitiveTypeUtils")),
        ("src/main/java/com/example/MyClass.java", ("com.example", "MyClass")),
        ("a/b/c/src/main/java/org/test/AnotherClass.java", ("org.test", "AnotherClass")),
        ("src/main/java/MyClass.java", ("", "MyClass")),  # No package
        ("src/main/java/com/example/MyClass.txt", (None, None)),  # not a java file
        ("src/main/java", (None, None)),  # No class file
        ("some/other/path/MyClass.java", (None, None)),  # Invalid path
        ("src/main/java/com/example/sub/MyClass.java", ("com.example.sub", "MyClass")),
        ("src/main/java/a/b/c/d/e/f/g/h/i/j/MyClass.java", ("a.b.c.d.e.f.g.h.i.j", "MyClass")),  # deeply nested package
        ("src/main/java/package-info.java", (None, None)),  # package-info.java should return None, None
        (123, (None, None)),  # handles non-string input
        ("src/main/java/a/b/c/MyClass.JAVA", (None, None)),  # handles uppercase JAVA
        ("src/main/java/a/b/c/MyClass.jAva", (None, None)),  # handles lowercase jAva
        ("src/main/java/a/b/c/MyClass.JaVa", (None, None)),  # handles mixed case JaVa
        ("src/main/java/a/b/c/MyClass.javA", (None, None)),  # handles mixed case javA

        # Test file paths
        ("src/test/java/com/example/MyTest.java", ("com.example", "MyTest")),
        ("src/test/java/org/test/AnotherTest.java", ("org.test", "AnotherTest")),
        ("a/b/c/src/test/java/org/test/DeepTest.java", ("org.test", "DeepTest")),
        ("src/test/java/NoPackageTest.java", ("", "NoPackageTest")),
        ("src/test/java/com/example/MyTest.txt", (None, None)), #not a java file
        ("src/test/java", (None, None)),  # No class file
        ("some/other/test/path/MyTest.java",(None,None)), # Invalid path
        ("src/test/java/a/b/c/d/e/f/g/h/i/j/MyTest.java", ("a.b.c.d.e.f.g.h.i.j", "MyTest")), #deeply nested package in test
        ("src/test/java/package-info.java", (None, None)),  # package-info.java should return None, None for test
        ("src/test/java/a/b/c/MyTest.JAVA", (None, None)),  # handles uppercase JAVA in test
        ("src/test/java/a/b/c/MyTest.jAva", (None, None)),  # handles lowercase jAva in test
        ("src/test/java/a/b/c/MyTest.JaVa", (None, None)),  # handles mixed case JaVa in test
        ("src/test/java/a/b/c/MyTest.javA", (None, None)),  # handles mixed case javA in test

    ]

    for file_path, expected_output in test_cases:
        output = extract_java_info(file_path)
        logger.info(f"File: {file_path}")
        logger.info(f"  Expected: {expected_output}")
        logger.info(f"  Actual:   {output}")
        if output == expected_output:
            logger.info("  Result: PASS")
        else:
            logger.info("  Result: FAIL")
        logger.info("-" * 20)
