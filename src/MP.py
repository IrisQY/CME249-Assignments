import numpy as np
from scipy.linalg import eig
import pandas as pd
from typing import TypeVar, Mapping, Set, Generic, Sequence

# Define MP by state set and transition matrix
# State set for MP: map (label to index)
# Transition matrix: matrix (nparray)
"""
    E.g.,
    state = {Rain: 0, Sunny: 1, Cloudy: 1, Windy: 1}
    tran_mat = np.asarray([0.1,0.2,0.3,0.4,
            0.25,0.25,0.25,0.25,
            0.1,0.2,0.3,0.4,
            0.25,0.25,0.25,0.25]).reshape((4,4))
    # Today's weather => tmr's weather
"""
class MP:

    # Initiate state dict & transition matrix
    def __init__(self, state: dict, tran_mat: np.array) -> None:
        # Check transition matrix and match state set with transition probs
        if np.sum(tran_mat, axis = 1) != np.ones(tran_mat.shape[0]):
            raise ValueError
        elif len(state) != tran_mat.shape[0]:
            raise ValueError
        else:
            self.state = state
            self.tran_mat = tran_mat

    # Get all states
    def get_states(self) -> set:
        return self.state.keys()

    # Compute stationary distribution using eigenvalue decomposition
    def stationary_dist(self) -> np.array:
        e_value, e_vec = eig(self.tran_mat,left=True,right=False)
        for num in e_value:
            if np.abs(num - 1.) < 1e-5:
                out = np.array(e_vec[:, num])
        out = out/np.sum(out)
        return out


