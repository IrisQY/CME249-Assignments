import numpy as np
import pandas as pd
from typing import TypeVar, Mapping, Set, Generic, Sequence
import MP

# Define MRP by state set and transition matrix
# State set for MRP: map (label to index)
# State with the 1st reward definition: map (label to reward)
# Transition matrix: matrix (nparray)
"""
    E.g.,
    state = {Rain: 0, Sunny: 1, Cloudy: 1, Windy: 1}
    tran_mat = np.asarray([0.1,0.2,0.3,0.4,
            0.25,0.25,0.25,0.25,
            0.1,0.2,0.3,0.4,
            0.25,0.25,0.25,0.25]).reshape((4,4))
    state_reward = {Rain: 1, Sunny: 2, Cloudy: 3, Windy: 4}
    # Today's weather => tmr's weather
"""
class MRP(MP):

    # Initiate state with reward and discount
    def __init__(self, state_reward: dict, gamma: float) -> None:
        if gamma <0 or gamma >1:
            raise ValueError
        else:
            self.reward = state_reward
            self.gamma = gamma
