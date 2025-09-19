import numpy as np
import functools

def binaryop_shape_match(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.operand1.shape != self.operand2.shape:
            raise ValueError(
                f"Shape mismatch in {func.__qualname__}: "
                f"{self.operand1.data.shape} vs {self.operand2.data.shape}"
            )
        return func(self, *args, **kwargs)
    return wrapper
