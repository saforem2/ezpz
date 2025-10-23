"""Simple test for the main ezpz module."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_import():
    """Test that ezpz can be imported."""
    try:
        import ezpz

        assert ezpz is not None
        print("✓ ezpz imported successfully")
    except Exception as e:
        print(f"✗ Failed to import ezpz: {e}")
        raise


def test_version():
    """Test that ezpz has a version."""
    import ezpz

    assert hasattr(ezpz, "__version__")
    assert isinstance(ezpz.__version__, str)
    assert len(ezpz.__version__) > 0
    print(f"✓ ezpz version: {ezpz.__version__}")


if __name__ == "__main__":
    test_import()
    test_version()
    print("All tests passed!")
