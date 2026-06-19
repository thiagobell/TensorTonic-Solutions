import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    if type(x) == list:
        x = np.array(x)
    if x is None:
        return None
    # Write code here
    return 1/(1+np.exp(-x))