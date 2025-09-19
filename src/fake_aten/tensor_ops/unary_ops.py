import numpy as np

def abs(operand: np.ndarray) -> np.ndarray:
    return np.abs(operand)

def neg(operand: np.ndarray) -> np.ndarray:
    return np.negative(operand)

def sign(operand: np.ndarray) -> np.ndarray:
    return np.sign(operand)

def exp(operand: np.ndarray) -> np.ndarray:
    return np.exp(operand)

def relu(operand: np.ndarray) -> np.ndarray:
    return np.maximum(0, operand)
