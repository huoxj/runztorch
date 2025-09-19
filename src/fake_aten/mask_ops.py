import numpy as np

def mask_gt(operand: np.ndarray, threshold) -> np.ndarray:
    return operand > threshold

def mask_lt(operand: np.ndarray, threshold) -> np.ndarray:
    return operand < threshold

def mask_ge(operand: np.ndarray, threshold) -> np.ndarray:
    return operand >= threshold

def mask_le(operand: np.ndarray, threshold) -> np.ndarray:
    return operand <= threshold

def mask_eq(operand: np.ndarray, threshold) -> np.ndarray:
    return operand == threshold

def mask_ne(operand: np.ndarray, threshold) -> np.ndarray:
    return operand != threshold

