"""
Unit and regression test for the mr_toolkit package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import mr_toolkit


def test_mr_toolkit_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "mr_toolkit" in sys.modules
