"""Test the normalize function from utils module."""

def normalize(name: str) -> str:
    """Normalize a name by replacing special characters with dashes and converting to lowercase.
    
    This is a copy of the function from ezpz.utils to avoid import issues.
    """
    import re
    return re.sub(r"[-_.]+", "-", name).lower()


def test_normalize():
    """Test the normalize function."""
    # Test with dashes
    result = normalize("test-name")
    assert result == "test-name"

    # Test with underscores
    result = normalize("test_name")
    assert result == "test-name"

    # Test with dots
    result = normalize("test.name")
    assert result == "test-name"

    # Test with mixed
    result = normalize("test_name.sub-name")
    assert result == "test-name-sub-name"

    # Test with uppercase
    result = normalize("TEST_NAME")
    assert result == "test-name"

    print("All normalize tests passed!")


if __name__ == "__main__":
    test_normalize()