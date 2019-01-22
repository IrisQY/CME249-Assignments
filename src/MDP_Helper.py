import numpy as np
import pandas as pd
from typing import TypeVar, Mapping, Set, Generic, Sequence

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

# Get actions helper function
def get_actions_helper(in_graph: dict) -> dict:
    state = get_states_helper(in_graph)
    actions_set = set()
    for s in state:
        temp_set = set(in_graph[s].keys())
        actions_set.update(temp_set)
    actions_list = list(actions_set)
    ind = range(len(actions_list))
    actions = dict(zip(actions_list,ind))
    return actions

# Get transition matrix helper function
def get_transition_helper(in_graph: dict) -> np.ndarray:
    states = get_states_helper(in_graph)
    actions = get_actions_helper(in_graph)
    tran_mat = np.zeros((len(states),len(actions),len(states))) # States * actions * states
    for i, row in in_graph.items():
        for c, action in row.items():
            for j, prob in action.items():
                ind_row = states[i]
                ind_height = states[j]
                ind_col = actions[c]
                if ind(tran_mat[ind_row,ind_col,ind_height],0):
                    tran_mat[ind_row,ind_col,ind_height] = prob
    return tran_mat

# Get reward matrix helper function
def get_reward_helper(in_graph: dict, state_action_reward: dict) -> np.ndarray:
    states = get_states_helper(in_graph)
    actions = get_actions_helper(in_graph)
    reward_mat = np.zeros((len(states),len(actions))) # States * actions
    for i, row in state_action_reward.items():
        for j, reward in row.items():
            ind_row = states[i]
            ind_col = actions[j]
            if ind(reward_mat[ind_row,ind_col],0):
                reward_mat[ind_row,ind_col] = reward
    return reward_mat

# Get policy matrix helper function
def get_policy_helper(in_graph: dict, policy: dict) -> np.ndarray:
    states = get_states_helper(in_graph)
    actions = get_actions_helper(in_graph)
    policy_mat = np.zeros((len(states),len(actions)))
    for i, row in policy.items():
        for j, prob in row.items():
            ind_row = states[i]
            ind_col = actions[j]
            if ind(policy_mat[ind_row,ind_col],0):
                policy_mat[ind_row,ind_col] = prob
    return policy_mat
