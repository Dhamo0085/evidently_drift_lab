"""Shared synthetic data. Unit tests never touch the network."""
import numpy as np
import pandas as pd
import pytest

NUM = ["x", "y"]
CAT = ["color"]


def make_frame(n: int, x_mean: float = 0.0, colors=("red", "blue"), color_p=None, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "x": rng.normal(x_mean, 1.0, n),
        "y": rng.normal(0.0, 1.0, n),
        "color": rng.choice(list(colors), n, p=color_p),
    })


@pytest.fixture
def same_large():
    """Reference/current from the same distribution, big enough for distance-based tests."""
    return make_frame(3000, seed=1), make_frame(3000, seed=2)


@pytest.fixture
def shifted_large():
    """Only x shifts (by one standard deviation)."""
    return make_frame(3000, seed=1), make_frame(3000, x_mean=1.0, seed=2)


@pytest.fixture
def adult_like():
    """Tiny stand-in for the Adult dataset, with the columns the scenarios use."""
    rng = np.random.default_rng(3)
    n = 400
    return pd.DataFrame({
        "age": rng.integers(18, 80, n),
        "education": rng.choice(["HS-grad", "Bachelors", "Some-college", "Masters", "Doctorate"], n),
        "hours-per-week": rng.integers(10, 60, n),
    })
