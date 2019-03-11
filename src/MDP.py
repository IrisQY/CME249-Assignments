import numpy as np
import pandas as pd
from typing import TypeVar, Mapping, Set, Generic, Sequence
from src.MRP import MRP
from src.MDP_Helper import get_states_helper, get_actions_helper, get_transition_helper_mdp, get_reward_helper,get_policy_helper
# Define MDP by Graph
"""
    E.g., 
    Input = {'H':
             {
              'Fast': {'H': 0.3, 'M': 0.3, 'L': 0.4},
              'Slow': {'H': 0.4, 'M': 0.3, 'L': 0.3}
             },
             'M':
             {
              'Fast': {'H': 0.2, 'M': 0.4, 'L': 0.4},
              'Slow': {'H': 0.4, 'M': 0.4, 'L': 0.2}
             },
             'L':
             {
              'Fast': {'H': 0.0, 'M': 0.2, 'L': 0.8},
              'Slow': {'H': 0.4, 'M': 0.4, 'L': 0.2}
             }}
    state_action_reward = {'H':{'Fast': 3, 'Slow': 2},
                           'M':{'Fast': 2, 'Slow': 2},
                           'L':{'Fast': 1, 'Slow': 3}}
    policy = {'H':{'Fast': 0.8, 'Slow': 0.2},
              'M':{'Fast': 0.5, 'Slow': 0.5},
              'L':{'Fast': 0.2, 'Slow': 0.8}}
    gamma = 0.5
    Meaning: Queuing example
"""
class MDP():

    def __init__(self, in_graph: dict, state_action_reward: dict, policy: dict, gamma: float) -> None:
        self.state: dict = get_states_helper(in_graph)
        self.action: dict = get_actions_helper(in_graph)
        self.tran_mat: np.ndarray = get_transition_helper_mdp(in_graph)
        self.reward: np.ndarray = get_reward_helper(in_graph,state_action_reward)
        self.policy: np.ndarray = get_policy_helper(in_graph,policy)
        self.gamma: float = gamma

    # Get all states
    def get_states(self) -> dict:
        return self.state

    # Get the transition matrix
    def get_tran_mat(self) -> np.ndarray:
        return self.tran_mat

    # Get all actions
    def get_actions(self) -> dict:
        return self.action

    # Get reward mat
    def get_reward(self) -> np.ndarray:
        return self.reward

    # Get policy mat
    def get_policy(self) -> np.ndarray:
        return self.policy

    # Get gamma
    def get_gamma(self) -> float:
        return self.gamma

    # Generate MRP
    def generate_MRP(self, policy: dict):
        print(self.reward)
        print(self.policy)
        R_S = dict(zip(list(self.state.keys()),[0]*len(self.state)))
        P_S = np.zeros((len(self.state),len(self.state)))
        in_graph = dict()
        for s_cur, i in self.state.items():
            tmp = 0
            value = dict()
            in_graph.update({s_cur:value})
            for a, j in self.action.items():
                R_S[s_cur] += self.reward[i,j]*self.policy[i,j]
                for s_next, c in self.state.items():
                    if s_next in value:
                        value[s_next] += self.tran_mat[i,j,c]*self.policy[i,j]
                    else:
                        value.update({s_next: self.tran_mat[i,j,c]*self.policy[i,j]})
        return MRP(in_graph,R_S,self.gamma)

    # Compute state value function v_{\pi}(s)
    def state_value_func(self, policy) -> float:
        MRP_tmp = self.generate_MRP(policy)
        return np.linalg.inv(np.identity(len(MRP_tmp.state))-MRP_tmp.gamma*MRP_tmp.tran_mat).dot(MRP_tmp.reward)

    # Compute action value function q_{\pi}(s,a)
    def action_value_func(self, policy) -> float:
        out = np.zeros((len(self.state),len(self.action)))
        for s_cur,i in self.state.items():
            for a,j in self.action.items():
                out[i,j] += self.reward[i,j]
                for s_next,c in self.state.items():
                    out[i,j] += self.gamma*self.tran_mat[i,j,c]*self.state_value_func(policy)[c]
        return out
