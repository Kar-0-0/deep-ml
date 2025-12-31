import numpy as np

def dynamic_tanh(x: np.ndarray, alpha: float, gamma: float, beta: float) -> list[float]:
    out = gamma * np.tanh(alpha * x) + beta

    return out