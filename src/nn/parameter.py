import numpy as np

from tensor.tensor import Tensor

class Parameter(Tensor):

    def __init__(self, data: np.ndarray):
        super().__init__(data, requires_grad=True)
