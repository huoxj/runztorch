import numpy as np

from autograd.funtion import Function
from tensor.tensor import Tensor
import fake_aten.tensor_ops as ato
import fake_aten as aten

class UnaryFunction(Function):
    def __init__(self, operand: Tensor):
        self.next_functions = [operand]

        self.operand = operand

class AbsOp(UnaryFunction):
    def __init__(self, operand: Tensor):
        super().__init__(operand)

    def forward(self) -> Tensor:
        result = Tensor(abs(self.operand.data))
        result.grad_fn = self
        return result

    def backward(self, *grad_outputs: np.ndarray) -> np.ndarray:
        (grad_output, ) = grad_outputs
        grad = ato.mul(grad_output, ato.sign(self.operand.data))
        return grad

class ExpOp(UnaryFunction):
    def __init__(self, operand: Tensor):
        super().__init__(operand)

    def forward(self) -> Tensor:
        result = Tensor(np.exp(self.operand.data))
        result.grad_fn = self
        return result

    def backward(self, *grad_outputs: np.ndarray) -> np.ndarray:
        (grad_output, ) = grad_outputs
        grad = ato.mul(grad_output, ato.exp(self.operand.data))
        return grad

class ReluOp(UnaryFunction):
    def __init__(self, operand: Tensor):
        super().__init__(operand)
    
    def forward(self) -> Tensor:
        result = Tensor(ato.relu(self.operand.data))
        result.grad_fn = self
        return result

    def backward(self, *grad_outputs: np.ndarray) -> np.ndarray:
        (grad_output, ) = grad_outputs
        grad = ato.mul(grad_output, aten.mask_gt(self.operand.data, 0))
        return grad
