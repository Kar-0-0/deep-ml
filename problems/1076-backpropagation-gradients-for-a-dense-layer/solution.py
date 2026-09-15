import numpy as np

def dense_backward(a_prev: np.ndarray, W: np.ndarray, b: np.ndarray, y: np.ndarray) -> dict:
    """
    Compute gradients of the squared-error cost w.r.t. W, b, and a_prev
    for a dense layer z = W @ a_prev + b followed by sigmoid activation.

    Args:
        a_prev: activations from previous layer, shape (K,)
        W: weight matrix, shape (J, K)
        b: bias vector, shape (J,)
        y: target vector, shape (J,)

    Returns:
        dict with keys 'dW' (J x K nested list), 'db' (list of length J),
        and 'da_prev' (list of length K).
    """
    z = W @ a_prev[:, None] + b[:, None] # (J, K) @ (K, 1) + (J, 1) --> (J, 1)
    simgoid = lambda x: 1 / (1 + np.e**(-x))
    sig = simgoid(z) # (J, 1)
    loss = np.power(sig - y, 2) # (J, 1)

    dl_dsig = 2 * (sig - y[:, None]) # (J, 1)
    dsig_dz = sig * (1 - sig) # (J, 1)
    dl_dz = dl_dsig * dsig_dz # (J, 1)

    # dl_dz @ dz/dw = dl_dw
    dz_dw = a_prev[:, None] # (K, 1)
    dl_dw = dl_dz @ dz_dw.T# (J, 1) @ (1, K)

    # dl_dz @ dz/da_prev = dl/da_prev
    dz_da_prev = W # (J, K) 
    dl_da_prev = dz_da_prev.T @ dl_dz # (K, J) @ (J, 1) --> (K, 1)
    dl_db = dl_dz # (J, 1)

    return {
        'dW' : dl_dw.tolist(),
        'db' : dl_db.flatten().tolist(),
        'da_prev' : dl_da_prev.flatten().tolist()  
    }