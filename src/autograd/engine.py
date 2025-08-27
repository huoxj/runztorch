import numpy as np

from tensor.tensor import Tensor

class Engine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def run_backward(self, tensor: Tensor, grad: np.ndarray):
        