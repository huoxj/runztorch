import numpy as np

from autograd.funtion import Function
from utils import wrap_data

class Tensor:
    def __init__(
        self, data, requires_grad = False
    ):
        self.data = wrap_data(data)
        self.grad_fn: 'Function | None' = None
        self.grad = None
        self.requires_grad = requires_grad

    def backward(self, gradient = None):
        if gradient is None:
            gradient = np.ones_like(self.data)
        
        

    def is_scalar(self):
        return self.data.ndim == 0

    def __add__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("Operand must be a Tensor")

