"""Simple demonstration that the test structure works."""

import unittest

class TestBasicStructure(unittest.TestCase):
    """Test that the basic test structure works."""
    
    def test_example(self):
        """A simple test to verify the test framework works."""
        self.assertEqual(1 + 1, 2)
        
    def test_string_operations(self):
        """Test basic string operations."""
        test_string = "hello world"
        self.assertIn("world", test_string)
        self.assertEqual(test_string.upper(), "HELLO WORLD")

if __name__ == "__main__":
    unittest.main()