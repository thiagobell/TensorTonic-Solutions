import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    # select the coluns for each row
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    probs_targets = y_pred[np.arange(y_pred.shape[0]), y_true]
    return -(np.log(probs_targets).sum()/y_true.shape[0])