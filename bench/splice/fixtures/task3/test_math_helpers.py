"""Tests for math_helpers. Fails until double exists."""
from math_helpers import double


def test_double():
    assert double(4) == 8
    assert double(0) == 0
