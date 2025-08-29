from typing import Any, List, Tuple

from tensor.tensor import Tensor

class Function:
    next_functions: List[Tensor]
    saved_tensors: Tuple[Any, ...] = ()
    countdown: int = 0
    
    def __init__(self):
        pass

    def forward(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def backward(self, *grad_outputs: Any):
        raise NotImplementedError
