import numpy as np

from autograd.funtion import Function
from autograd.engine import Engine
from utils import wrap_data

class Tensor:
    data: np.ndarray
    grad: np.ndarray | None
    grad_fn: 'Function | None'
    requires_grad: bool

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
        
        Engine().run_backward(self, gradient)

    def is_scalar(self):
        return self.data.ndim == 0

    def __add__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("Operand must be a Tensor")

