"""Shared pytest fixtures for the NexaQuery test suite."""

import pandas as pd
import pytest


@pytest.fixture
def empty_dataset() -> pd.DataFrame:
    """Empty DataFrame — useful for shape assertions and trivial paths."""
    return pd.DataFrame()
