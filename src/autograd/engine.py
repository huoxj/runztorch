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
        self._backward_grad(tensor, grad)

    @staticmethod
    def _backward_countdown(root_fn: Function):
        queue = [root_fn]
        while len(queue) > 0:
            fn = queue.pop(0)
            for next_tensor in fn.next_functions:
                if next_tensor is None or next_tensor.grad_fn is None:
                    continue
                next_tensor.grad_fn.countdown += 1
                queue.append(next_tensor.grad_fn)

    @staticmethod
    def _backward_grad(root_tensor: Tensor, root_grad: np.ndarray):
        root_tensor.grad = root_grad
        if root_tensor.grad_fn is None:
            return
        
        queue: List[Tuple[Function, np.ndarray]] \
            = [(root_tensor.grad_fn, root_grad)]

        while len(queue) > 0:
            fn, grad = queue[0]

            grad_outputs = fn.backward(grad)

            if not isinstance(grad_outputs, tuple):
                grad_outputs = (grad_outputs,)
            
            if len(fn.next_functions) != len(grad_outputs):
                raise ValueError(f"Grad outputs length mismatch next functions length: expect {len(fn.next_functions)}, actual {len(grad_outputs)}")

            for next_tensor, grad_output in zip(fn.next_functions, grad_outputs):           
                if next_tensor is None:
                    continue
                    
                apply_grad(next_tensor, grad_output)

                next_fn = next_tensor.grad_fn
                if next_fn is None:
                    continue

                next_fn.countdown -= 1
                if next_fn.countdown == 0:
                    queue.append((next_fn, grad_output))
