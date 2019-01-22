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
