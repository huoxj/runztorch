import numpy as np

def wrap_data(data) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, (int, float)):
        return np.array(data, dtype=float)

    raise TypeError(f"Unsupported data type '{type(data).__name__}' for tensor initialization")
