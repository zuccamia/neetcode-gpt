import numpy as np
from numpy.typing import NDArray


class Solution:
    # Turns out Numpy's maths operations apply to entire arrays
    # at once, instead of iterating each scalar element!

    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)

        e = np.exp(-z)
        activation = 1 / (1 + e)
        return np.round(activation, 5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise

        return np.maximum(0,z)
