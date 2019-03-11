from typing import TypeVar, Mapping, Set, Callable, Tuple, Generic
import random

S = TypeVar('S')
A = TypeVar('A')

class MDPforFA():
    
    def __init__(self, state_action_func: Callable[[S], Set[A]], state_reward_func: Callable[[S, A], Tuple[S, float]],
                 terminal_states_func: Callable[[S], bool],init_state_func: Callable[[], S],
                 init_act_func: Callable[[S], A], gamma: float) -> None:
        self.state_action_func = state_action_func
        self.is_terminal_state = terminal_states_func
        self.state_reward_func = state_reward_func
        self.init_state_func = init_state_func
        self.init_act_func = init_act_func
        self.gamma = gamma

        
    def init_state_gen(self):
        s = self.init_state_func()
        return s

    def init_sa(self):
        s = self.init_state_func()
        a = self.init_act_func(s)
        return s, a

    def state_reward_func(self,s_cur,a_cur):
        s, r = self.state_reward_func(s_cur,a_cur)
        return s, r

    def is_terminal_state(self,s_cur):
        return 0
