"""
Test that all example files can be imported successfully.

This ensures that example files are compatible with the current package structure
and don't have any import errors.
"""

import unittest
import importlib.util
import sys
from pathlib import Path


class TestExamples(unittest.TestCase):
    """Test that all example Python files can be imported without errors."""

    def setUp(self):
        """Set up the test by finding all example files."""
        # Navigate from tests/ to examples/
        self.examples_dir = Path(__file__).parent.parent.parent / "examples"
        self.example_files = list(self.examples_dir.glob("*.py"))

    def test_example_imports(self):
        """Test that all example files can be imported successfully."""
        # Temporarily disabled - examples will be converted to notebooks for Google Colab
        self.skipTest("Examples test temporarily disabled pending notebook conversion")

        failed_imports = []

        for example_file in self.example_files:
            # Skip __init__.py if it exists
            if example_file.name == "__init__.py":
                continue

            module_name = f"examples.{example_file.stem}"

            try:
                # Try to import the module
                if module_name in sys.modules:
                    # Reload if already imported
                    importlib.reload(sys.modules[module_name])
                else:
                    __import__(module_name)

            except Exception as e:
                failed_imports.append((example_file.name, str(e)))

        # Report all failures at once
        if failed_imports:
            error_msg = "The following example files failed to import:\\n"
            for filename, error in failed_imports:
                error_msg += f"  - {filename}: {error}\\n"
            self.fail(error_msg)

    def test_individual_examples(self):
        """Test each example file individually for better error reporting."""
        for example_file in self.example_files:
            if example_file.name == "__init__.py":
                continue

            with self.subTest(example=example_file.name):
                module_name = f"examples.{example_file.stem}"

                try:
                    if module_name in sys.modules:
                        importlib.reload(sys.modules[module_name])
                    else:
                        __import__(module_name)

                except ImportError as e:
                    self.fail(f"Failed to import {example_file.name}: {e}")
                except Exception as e:
                    # Some examples might fail with other errors if they try to run code
                    # but we only care about import errors for this test
                    if "No module named" in str(e) or "cannot import" in str(e):
                        self.fail(f"Import error in {example_file.name}: {e}")
                    # Otherwise, it's likely a runtime error which is acceptable for this test


if __name__ == "__main__":
    unittest.main()
