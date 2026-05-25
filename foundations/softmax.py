import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        max_z = max(z)
        stable_z = [logit - max_z for logit in z]
        counts = [np.exp(logit) for logit in stable_z]
        denom = sum(counts)
        out = [value / denom for value in counts]
        return np.round(out, 4)

        
