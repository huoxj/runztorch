import numpy as np

from autograd.functions import Function
from tensor.tensor import Tensor

class ExpandOp(Function):
    def __init__(self, operand: Tensor, shape: tuple):
        self.next_functions = [operand]
        self.operand = operand
        self.shape = shape

    def forward(self) -> np.ndarray:
        result = np.broadcast_to(self.operand.data, self.shape)
        return result

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad = np.sum(grad_output, axis=tuple(i for i, (s1, s2) in enumerate(zip(grad_output.shape, self.operand.data.shape)) if s1 != s2), keepdims=True)
        return grad
