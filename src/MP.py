import numpy as np
import pandas as pd
import scipy
from typing import TypeVar,Mapping, Set, Generic, Sequence

# Helper functions
T = TypeVar("T",str,int,float)

# Identity helper function for str, int and float
def ind(x: T, y: T):
    if x == y or np.abs(x-y)<1e-5:
        return True
    else:
        return False

# Get state helper function
def get_states_helper(in_graph: dict) -> dict:
    state_list = list(in_graph.keys())
    ind = range(len(state_list))
    state = dict(zip(state_list,ind))
    return state

# Get transition matrix helper function
def get_transition_helper(in_graph: dict) -> np.ndarray:
    state = get_states_helper(in_graph)
    tran_mat = np.zeros((len(state),len(state)))
    for i, row in in_graph.items():
        for j, prob in row.items():
            ind_row = state[i]
            ind_col = state[j]
            if ind(tran_mat[ind_row,ind_col],0):
                tran_mat[ind_row,ind_col] = prob
    return tran_mat

# Define MP by Graph
"""
    E.g.,
    Input = {'Sunny': {'Sunny': 0.1, 'Cloudy': 0.2, 'Rainy': 0.3, 'Cloudy': 0.4},
             'Cloudy': {'Sunny': 0.25, 'Cloudy': 0.25, 'Rainy': 0.3, 'Cloudy': 0.2},
             'Rainy': {'Sunny': 0.1, 'Cloudy': 0.2, 'Rainy': 0.3, 'Cloudy': 0.4},
             'Windy': {'Sunny': 0.25, 'Cloudy': 0.25, 'Rainy': 0.25, 'Cloudy': 0.25}}
    Meaning: Today's weather => tmr's weather
"""
class MP:
    # Initiate state dict & transition matrix
    def __init__(self, in_graph: dict) -> None:
        self.graph = in_graph
        state = get_states_helper(in_graph)
        tran_mat = get_transition_helper(in_graph)
        # Check transition matrix and match state set with transition probs
        if np.linalg.norm(np.sum(tran_mat, axis = 1)- np.ones(tran_mat.shape[0]))>1e-5:
            raise ValueError
        elif len(state) != tran_mat.shape[0]:
            raise ValueError
        else:
            self.state: dict = state
            self.tran_mat: np.ndarray = tran_mat

    # Get all states
    def get_states(self) -> set:
        return self.state

    # Get the transition matirx
    def get_tran_mat(self) -> np.ndarray:
        return self.tran_mat

    # Compute stationary distribution using eigenvalue decomposition
    def stationary_dist(self) -> np.array:
        e_value, e_vec = np.linalg.eig(self.tran_mat.T)
        out = np.array(e_vec[:, np.where(np.abs(e_value- 1.) < 1e-5)[0][0]])
        out = out/np.sum(out)
        return out
