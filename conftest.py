"""Pytest configuration shared by all tests."""

from __future__ import annotations


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "gpu: needs a real FP8 GPU + checkpoint (slow; skipped by `pytest -m 'not gpu'`)",
    )
