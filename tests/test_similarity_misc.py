import numpy as np

from utils.similarity import numpy_cosine_similarity


def test_numpy_cosine_similarity_basic():
    v1 = np.array([1.0, 0.0])
    v2 = np.array([1.0, 0.0])
    assert numpy_cosine_similarity(v1, v2) == 1.0


def test_numpy_cosine_similarity_none():
    assert numpy_cosine_similarity(None, None) == 0.0


def test_numpy_cosine_similarity_shape_mismatch():
    assert numpy_cosine_similarity(np.array([1.0, 0.0]), np.array([1.0])) == 0.0
