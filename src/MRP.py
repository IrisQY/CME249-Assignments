import numpy as np
import pandas as pd
from typing import TypeVar, Mapping, Set, Generic, Sequence
import MP

# Convert reward helper function
def convert_reward(_2nd_def_reward: dict, tran_mat: np.ndarray, state: dict) -> dict:
    reward_mat = np.zeros((len(state),len(state)))
    # Create reward matrix
    for i, row in _2nd_def_reward.items():
        for j, reward in row.items():
            ind_row = state[i]
            ind_col = state[j]
            if MP.ind(reward[ind_row,ind_col],0):
                reward[ind_row,ind_col] = reward
    # Cast to 1st def reward vector
    reward_vec = np.diag(tran_mat.dot(reward_mat.T))
    reward_dict = dict(zip(state.keys(),reward_vec))
    return reward_dict

# Define MRP by Graph
"""
    E.g.,
    Input = {'Sunny': {'Sunny': 0.1, 'Cloudy': 0.2, 'Rainy': 0.3, 'Cloudy': 0.4},
             'Cloudy': {'Sunny': 0.25, 'Cloudy': 0.25, 'Rainy': 0.3, 'Cloudy': 0.2},
             'Rainy': {'Sunny': 0.1, 'Cloudy': 0.2, 'Rainy': 0.3, 'Cloudy': 0.4},
             'Windy': {'Sunny': 0.25, 'Cloudy': 0.25, 'Rainy': 0.25, 'Cloudy': 0.25}}
    state_reward = {'Rain': 1, 'Sunny': 2, 'Cloudy': 3, 'Windy': 4}
    gamma = 0.5
    Meaning: Today's weather => tmr's weather
"""
class MRP(MP):

    # Initiate state with reward and discount
    def __init__(self, state_reward: dict, gamma: float) -> None:
        if gamma <0 or gamma >1:
            raise ValueError
        else:
            reward_vec = np.zeros(len(self.state))
            for key, ind in self.state.items():
                reward_vec[ind] = state_reward[key]
            self.reward: np.ndarray = reward_vec
            self.gamma: float = gamma

    # Compute value function R(s)
    def value_func(self) -> float:
        return np.linalg.inv(np.identity(len(self.state))-self.gamma*self.tran_mat).dot(self.reward)

    # Compute value function r(s,s')
    def value_func_2nd(self,_2nd_def_reward) -> float:
        reward_dict = convert_reward(_2nd_def_reward)
        reward_vec = np.zeros(len(self.state))
        for key, ind in self.state.items():
            reward_vec[ind] = reward_dict[key]
        self.reward = reward_vec
        return self.value_func()
