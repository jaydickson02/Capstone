import numpy as np


def normalise(vector):
    """Normalise a vector to unit length."""
    return vector / np.linalg.norm(vector)


def magnitude(vector):
    """Return the magnitude of a vector."""
    return np.linalg.norm(vector)
