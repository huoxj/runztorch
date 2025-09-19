import numpy as np

def add(operand1: np.ndarray, operand2: np.ndarray) -> np.ndarray:
    return np.add(operand1, operand2)

def sub(operand1: np.ndarray, operand2: np.ndarray) -> np.ndarray:
    return np.subtract(operand1, operand2)

def mul(operand1: np.ndarray, operand2: np.ndarray) -> np.ndarray:
    return np.multiply(operand1, operand2)

def div(operand1: np.ndarray, operand2: np.ndarray) -> np.ndarray:
    return np.divide(operand1, operand2)

def mmul(operand1: np.ndarray, operand2: np.ndarray) -> np.ndarray:
    return np.matmul(operand1, operand2)
