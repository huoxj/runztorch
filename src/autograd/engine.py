from typing import List, Tuple
import numpy as np

from autograd.funtion import Function
from tensor.tensor import Tensor

def apply_grad(tensor: Tensor, grad: np.ndarray):
    if tensor.grad is None:
        tensor.grad = grad
        return

    if tensor.grad.shape != grad.shape:
        raise ValueError(f"Grad shape mismatch: expect {tensor.grad.shape}, actual {grad.shape}")

    tensor.grad = tensor.grad + grad

def handle_func_backward(fn: Function, grad: np.ndarray):
    pass

class Engine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def run_backward(self, tensor: Tensor, grad: np.ndarray):
        if tensor.grad_fn is None:
            return

        self._backward_countdown(tensor.grad_fn)
        

    @staticmethod
    def _backward_countdown(root_fn: Function):
        root_fn.countdown = 0
        queue = [root_fn]
        while len(queue) > 0:
            fn = queue.pop(0)
            for next_tensor in fn.next_functions:
                if next_tensor is None or next_tensor.grad_fn is None:
                    continue
                next_tensor.grad_fn.countdown += 1
                queue.append(next_tensor.grad_fn)

    @staticmethod
    def _backward_grad(root_tensor: Tensor, grad: np.ndarray):
        root_tensor.grad = grad
        if root_tensor.grad_fn is None:
            return
        
        queue: List[Tuple[Function, np.ndarray]] \
            = [(root_tensor.grad_fn, grad)]

        while len(queue) > 0:
            fn = queue.pop(0)
            
