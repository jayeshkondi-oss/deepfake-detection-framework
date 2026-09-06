import sys
import platform


def test_environment():
    """Verify that the Python environment is working."""

    print("=" * 60)
    print("DEEPFAKE DETECTION FRAMEWORK")
    print("=" * 60)

    print(f"Python      : {sys.version}")
    print(f"Platform    : {platform.platform()}")
    print(f"Architecture: {platform.machine()}")

    assert sys.version_info >= (3, 11)

    print("\nEnvironment test: PASSED")