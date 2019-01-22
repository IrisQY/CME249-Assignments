import numpy as np
import pandas as pd
from typing import TypeVar, Mapping, Set, Generic, Sequence
import MP
from MRP_Helper import convert_reward

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
    def __init__(self, in_graph: dict, state_reward: dict, gamma: float) -> None:
        super().__init__(in_graph)
        self.state = self.get_states()
        self.tran_mat = self.get_tran_mat()
        if gamma <0 or gamma >1:
            raise ValueError
        else:
            reward_vec = np.zeros(len(self.state))
            for key, ind in self.state.items():
                reward_vec[ind] = state_reward[key]

            self.reward: np.ndarray = reward_vec
            self.gamma: float = gamma

    # Get all states
    def get_states(self) -> set:
        return self.state

    # Get the transition matirx
    def get_tran_mat(self) -> np.ndarray:
        return self.tran_mat

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
