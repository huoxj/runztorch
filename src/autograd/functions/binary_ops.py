from typing import Tuple
import numpy as np

from autograd.funtion import Function
from tensor.tensor import Tensor
from utils import unbroadcast

class BinaryFunction(Function):
    def __init__(self, operand1: Tensor, operand2: Tensor):
        self.next_functions = [operand1, operand2]

        self.operand1 = operand1
        self.operand2 = operand2

class AddOp(BinaryFunction):
    def __init__(self, operand1: Tensor, operand2: Tensor):
        super().__init__(operand1, operand2)

    def forward(self) -> Tensor:
        result = Tensor(self.operand1.data + self.operand2.data)
        result.grad_fn = self
        return result

    def backward(self, *grad_outputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        (grad_output, ) = grad_outputs
        grad1 = unbroadcast(grad_output, self.operand1.data.shape)
        grad2 = unbroadcast(grad_output, self.operand2.data.shape)
        return grad1, grad2

class SubOp(BinaryFunction):
    def __init__(self, operand1: Tensor, operand2: Tensor):
        super().__init__(operand1, operand2)

    def forward(self) -> Tensor:
        result = Tensor(self.operand1.data - self.operand2.data)
        result.grad_fn = self
        return result

    def backward(self, *grad_outputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        (grad_output, ) = grad_outputs
        grad1 = unbroadcast(grad_output, self.operand1.data.shape)
        grad2 = unbroadcast(-grad_output, self.operand2.data.shape)
        return grad1, grad2

class MulOp(BinaryFunction):
    def __init__(self, operand1: Tensor, operand2: Tensor):
        self.next_functions = [operand1, operand2]
        self.saved_tensors = ()

        self.operand1 = operand1
        self.operand2 = operand2

    def forward(self) -> Tensor:
        result = Tensor(self.operand1.data * self.operand2.data)
        result.grad_fn = self
        return result

    def backward(self, *grad_outputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        (grad_output, ) = grad_outputs
        grad1 = unbroadcast(grad_output * self.operand2.data, self.operand1.data.shape)
        grad2 = unbroadcast(grad_output * self.operand1.data, self.operand2.data.shape)
        return grad1, grad2

class DivOp(BinaryFunction):
    def __init__(self, operand1: Tensor, operand2: Tensor):
        self.next_functions = [operand1, operand2]
        self.saved_tensors = ()

        self.operand1 = operand1
        self.operand2 = operand2

    def forward(self) -> Tensor:
        result = Tensor(self.operand1.data / self.operand2.data)
        result.grad_fn = self
        return result

    def backward(self, *grad_outputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        (grad_output, ) = grad_outputs
        grad1 = unbroadcast(grad_output / self.operand2.data, self.operand1.data.shape)
        grad2 = unbroadcast(-grad_output * self.operand1.data / (self.operand2.data ** 2), self.operand2.data.shape)
        return grad1, grad2
