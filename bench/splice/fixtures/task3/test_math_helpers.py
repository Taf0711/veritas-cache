"""Tests for math_helpers. The double test fails until the task is done."""
from math_helpers import add, double


def test_add():
    assert add(2, 3) == 5


def test_double():
    assert double(4) == 8
    assert double(0) == 0
