"""Tests for text_utils. The empty input test fails until the task is done."""
from text_utils import first_word


def test_first_word_normal():
    assert first_word("hi there") == "hi"


def test_first_word_empty():
    assert first_word("") == ""
