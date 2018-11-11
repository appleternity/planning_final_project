from copy import copy, deepcopy
import random
from util import Counter, Array, flipcoin, parsing_config
from display import Display
import os.path
import json
import numpy as np

class State:
    _N = None
    _GRAGH = None

    @staticmethod
    def set_value(n, graph):
        State._N = n
        State._GRAGH = graph


    def __init__(self, p=None, c=None):
        """
        :param p: tuple of int, position for all pursuer. e.g: (1,1,1,2,3)
        :param c: tuple of int, clean: 0, dirty:1 e.g: (0,1,1,1,1)
        """
        # keep: pursuer position, contaminated region

        self.pursuers =  tuple(0 for _ in range(State._N)) if p is None else p
        self.dirty = (0,) + tuple(1 for _ in range(len(State._GRAGH) - 1)) if c is None else c

    def __hash__(self):
        return hash((self.pursuers, self.dirty))

    def __str__(self):
        return "State({}, {})".format(str(self.pursuers), str(self.dirty))

    def contaminate(self, p_old, p_new, dirty_old):
        """
        return: tuple(new_dirty)
        """

        dirty = dirty_old
        new_dirty = list(dirty[:])
        move_pos = None
        for p1, p2 in zip(p_old, p_new):
            if p1 != p2:
                move_pos = p1
                new_dirty[p2] = 0
                new_dirty[p1] = 0
                break
        
        if move_pos in p_new:
            return dirty

        stack = [move_pos]
        visited = set()

        while stack:
            node = stack.pop()

            # contaminate node if any neighbor is contaminated
            if any([new_dirty[nei] for nei in State._GRAGH[node]]):
                # new_dirty = new_dirty[:node] + (1,) + new_dirty[node + 1:]
                new_dirty[node] = 1
                for nei in State._GRAGH[node]:
                    if nei in p_new or nei in visited:
                        continue
                    else:
                        visited.add(nei)
                        if not dirty[nei]:
                            stack.append(nei)
                        new_dirty[nei] = 1

        return tuple(new_dirty)

    def get_legal_action(self):
        """
        return [action,action, ... ...]
        """
        pursuers, dirty = self.pursuers, self.dirty
        #next_pursuers = set()
        action_set = set()

        for i in range(len(pursuers)):
            for p in State._GRAGH[pursuers[i]]:
                a = Action(i, p)
                if a not in action_set:
                    action_set.add(a)

        return [i for i in action_set]

    def get_successors(self):
        """
        return [(next_state, action), (next_state, action), ...]
        """

        pursuers, dirty = self.pursuers, self.dirty
        next_state = set()
        next_pursuers = set()

        for i in range(len(pursuers)):
            for p in State._GRAGH[pursuers[i]]:
                next_p = (pursuers[:i] + (p,) + pursuers[i + 1:], Action(i, p))

                if next_p not in next_pursuers:
                    next_pursuers.add(next_p)

        for (next_ps, action) in next_pursuers:
            for origin_p, next_p in zip(pursuers, next_ps):
                if origin_p != next_p:
                    new_dirty = dirty[:next_p] + (0,) + dirty[next_p + 1:]
                    if origin_p not in next_ps:
                        new_dirty = self.contaminate(origin_p, next_ps, new_dirty)
                    new_state = State(next_ps, new_dirty)

                    next_state.add( (new_state, action) )

        return [i for i in next_state]

    def is_goal(self):
        # return True/False
        return sum(self.dirty) == 0

    def do_action(self, action):
        """
        return next_state
        """

        # potentially double compute contaminate !!

        # update cur state, contaminate
        # next_ps = ori [pid]-> next
        #print(action)
        next_ps = self.pursuers[:action.p_id] + (action.next_node_id,) + self.pursuers[action.p_id + 1:]
        #new_dirty = self.dirty[:action.p_id] + (0,) + self.dirty[action.p_id + 1:]
        new_dirty = self.contaminate(self.pursuers, next_ps, self.dirty)

        self.pursuers = next_ps
        self.dirty = new_dirty

    def deep_copy(self):
        """
        return a clone of self
        """
        return State(self.pursuers, self.dirty)

    # def reset(self):
    #     pass

class Action:
    def __init__(self, p_id, next_node_id):
        # keep: (p_id, next_node_id)
        self.p_id = p_id
        self.next_node_id = next_node_id

    def __hash__(self):
        return hash((self.p_id, self.next_node_id))

    def __str__(self):
        return "Action({}, {})".format(str(self.p_id), str(self.next_node_id))

class Agent:

    def __init__(self, alpha=1.0, epsilon=0.05, gamma=0.8, num_episode=1000):
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.discount = float(gamma)
        self.q_value = Counter()
        self.eposodes = 0
        self.total_training_rewards = 0.0
        self.num_episode = 1000

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def start_episode(self):
        self.last_state = None
        self.last_action = None
        self.episode_rewards = 0.0

    def observe_transition(self, state, action, next_state, delta_reward):
        #self.episode_rewards += delta_reward
        self.update(state, action, next_state, delta_reward)

    # TODO: check hash
    def get_qvalue(self, state, action):
        return self.q_value[(hash(state), hash(action))]

    def compute_value_from_qvalue(self, state):
        try:
            return max(self.get_qvalue(state, action) for action in state.get_legal_action())
        except ValueError:
            return 0.0

    def compute_action_from_qvalue(self, state):
        try:
            return max([
                (self.get_qvalue(state, action), action)
                for action in state.get_legal_action()
            ], key=lambda x:x[0])[1]
        except ValueError:
            return None

    def get_action(self, state):
        legal_actions = state.get_legal_action()
        if not legal_actions: return action

        if flipcoin(self.epsilon):
            action = random.choice(legal_actions)
        else:
            action = self.compute_action_from_qvalue(state)

        return action

    def update(self, state, action , next_state, reward):
        #self.q_value[(state, action)] += self.alpha * (
        #    reward + self.discount * self.get_value(next_state) - self.get_qvalue(state, action))
        next_value = self.get_value(next_state)
        self.q_value[(hash(state), hash(action))] = sum([
            (1-self.alpha) * self.get_qvalue(state, action),
            self.alpha * (reward + self.discount*next_value)
        ])

    def get_policy(self, state):
        return self.compute_action_from_qvalue(state)

    def get_value(self, state):
        return self.compute_value_from_qvalue(state)

class Environment:
    def __init__(self, num_p, map_type, k, w):
        # num_p, graph
        # A = State(n, graph)
        self.num_p = num_p
        filename = "_{}_k{}_w{}".format(map_type, k, w)

        # load graph
        with open(os.path.join("state", filename+"_state.json"), 'r') as infile:
            graph = json.load(infile)
            graph = {int(k):v for k, v in graph.items()}

        with open(os.path.join("mapping", filename+"_mapping_dictionary.json"), 'r') as infile:
            mapping = json.load(infile)
            mapping = {int(k):(int(v[0]), int(v[1])) for k, v in mapping.items()}

        conf = parsing_config()[map_type]["{},{}".format(k, w)]
        State.set_value(num_p, graph)
        self.display = Display(graph, mapping, map_type, k, w, 
                fix_r=int(conf["fix_r"]), fix_c=int(conf["fix_c"]), unit=int(conf["unit"]))
        self.state = State()
        self.state_list = []

        # reward setting
        self.step_limit = 150
        self.step = 0

    def get_current_state(self):
        return self.state

    def reset(self):
        self.state = State()
        self.step = 0
        self.state_list.clear()

    def is_goal(self):
        if self.state.is_goal():
            return (True, True)
        if self.step == self.step_limit:
            return (True, False)
        return (False, False)

    def reward(self):
        # version 1: number of clean region
        if self.state.is_goal():
            return 100
        if self.step == self.step_limit:
            return -50
        return sum(1 for d in self.state.dirty if d == 0) - self.step*0.3

    def get_possible_actions(self):
        return self.state.get_legal_action()

    def display_state(self):
        print("\rstep={}".format(self.step), end="   ")
        self.display.draw(self.state, 10)

    def update_state_list(self, state):
        self.state_list.append(state)

    def do_action(self, action):
        self.state.do_action(action)
        self.step += 1
        return self.state, self.reward()

    def replay(self):
        for s in self.state_list:
            self.display.draw(s, 30)

    def training(self):
        pass

def run_episode(env, e, show, agent, discount):
    returns = 0
    totalDiscount = 1.0
    env.reset()

    while True:
        state = env.get_current_state().deep_copy()
        env.update_state_list(state)

        if show:
            env.display_state()
        
        # check goal
        g1, g2 = env.is_goal()
        if g1:
            if g2:
                print("clean!!!!!!!!!")
                env.replay()
                #quit()
            return returns
        
        # make decision - choose action
        action = agent.get_action(state) 

        # do action
        next_state, reward = env.do_action(action)

        # update
        agent.observe_transition(state, action, next_state, reward)

        returns += reward * totalDiscount
        totalDiscount *= discount
        #print(state, reward, next_state, action)

    print("start running")

def training():
    discount = 0.8
    num_p = 2
    map_type = "tree"
    k = 1
    w = 1
    """
    num_p = 2
    map_type = "ladder"
    k = 2
    w = 1
    """
    """
    num_p = 3
    map_type = "ladder"
    k = 3
    w = 1
    """
    """
    num_p = 3
    map_type = "ladder"
    k = 4
    w = 1
    """
    """
    num_p = 3
    map_type = "tree"
    k = 1
    w = 2
    """
    epsilon = 0.3
    learning_rate = 0.05
    agent = Agent(gamma=discount, epsilon=epsilon, alpha=learning_rate)
    env = Environment(num_p, map_type, k, w)
    array = Array(100, np.float32)
    for e in range(1, 20001):
        returns = run_episode(env, e, e%500==0, agent, discount)
        array.append(returns)
        #if e % 50 == 0:
        if e % 2000 == 0:
            epsilon *= 0.5
            agent.set_epsilon(epsilon)
            #with open("test.json", 'w', encoding='utf-8') as outfile:
            #    for k, v in agent.q_value.items():
            #        outfile.write("{} => {}\n".format(str(k), str(v)))
        print("\r e={}, epi={:.4f} returns={}, num={}".format(e, epsilon, array.average(), len(agent.q_value)), end="       ")

def main():
    pass

if __name__ == "__main__":
    training()
