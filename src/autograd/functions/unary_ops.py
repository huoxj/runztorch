import numpy as np

from autograd.funtion import Function
from tensor.tensor import Tensor

class UnaryFunction(Function):
    def __init__(self, operand: Tensor):
        self.next_functions = [operand]

        self.operand = operand

class ExpOp(UnaryFunction):
    def __init__(self, operand: Tensor):
        super().__init__(operand)

    def forward(self) -> Tensor:
        result = Tensor(np.exp(self.operand.data))
        result.grad_fn = self
        return result

    def backward(self, *grad_outputs: np.ndarray) -> np.ndarray:
        (grad_output, ) = grad_outputs
        grad = grad_output * np.exp(self.operand.data)
        return grad

class ReluOp(UnaryFunction):
    def __init__(self, operand: Tensor):
        super().__init__(operand)
    
    def forward(self) -> Tensor:
        result = Tensor(np.maximum(0, self.operand.data))
        result.grad_fn = self
        return result

    def backward(self, *grad_outputs: np.ndarray) -> np.ndarray:
        (grad_output, ) = grad_outputs
        grad = grad_output * (self.operand.data > 0)
        return grad
