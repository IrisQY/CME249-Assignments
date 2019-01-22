import numpy as np
import pandas as pd
import scipy
from typing import TypeVar,Mapping, Set, Generic, Sequence
from MP_Helper import ind

# Convert reward helper function
def convert_reward(_2nd_def_reward: dict, tran_mat: np.ndarray, state: dict) -> dict:
    reward_mat = np.zeros((len(state),len(state)))
    # Create reward matrix
    for i, row in _2nd_def_reward.items():
        for j, reward in row.items():
            ind_row = state[i]
            ind_col = state[j]
            if ind(reward_mat[ind_row,ind_col],0):
                reward_mat[ind_row,ind_col] = reward
    # Cast to 1st def reward vector
    reward_vec = np.diag(tran_mat.dot(reward_mat.T))
    reward_dict = dict(zip(state.keys(),reward_vec))
    return reward_dict
