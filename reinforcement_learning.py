from copy import copy, deepcopy
import random
from util import Counter, Array, flipcoin, parsing_config
from display import Display
import os, os.path
import json
import numpy as np
import pickle
from pprint import pprint

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
        #return hash((tuple(sorted(self.pursuers)), self.dirty))
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
        next_ps = self.pursuers[:action.p_id] + (action.next_node_id,) + self.pursuers[action.p_id + 1:]
        new_dirty = self.contaminate(self.pursuers, next_ps, self.dirty)

        self.pursuers = next_ps
        self.dirty = new_dirty

    def deep_copy(self):
        return State(self.pursuers, self.dirty)

    def clear_region(self):
        return sum(1 for d in self.dirty if d == 0)

class StateOne:
    _N = None
    _GRAGH = None
    
    @staticmethod
    def set_value(n, graph):
        StateOne._N = n
        StateOne._GRAGH = graph
    
    def __init__(self, g=None):
        self.g = g if g is not None else [StateOne._N if i == 0 else -1 for i, _ in enumerate(StateOne._GRAGH)]

    def __hash__(self):
        return hash(tuple(self.g))

    def __str__(self):
        return "StateOne({})".format(str(self.g))
    
    def is_goal(self):
        return sum(1 for g in self.g if g == -1) == 0

    def deep_copy(self):
        return StateOne(list(self.g))

    def get_legal_action(self): 
        action_set = set()
        for i, g in enumerate(self.g):
            if g >= 1:
                for p in StateOne._GRAGH[i]:
                    action_set.add(ActionOne(i, p))
        return [a for a in action_set]

    def do_action(self, action):
        if self.g[action.next_node_id] == -1:
            self.g[action.next_node_id] = 1
        else:
            self.g[action.next_node_id] += 1
        
        self.g[action.node_id] -= 1 # >= 1
        if self.g[action.node_id] == 0:
            # update dirty
            stack = [action.node_id]
            visited = set()
            alive = set()
            while stack:
                node = stack.pop()
                visited.add(node)
                if any(True if self.g[nei]==-1 else False for nei in StateOne._GRAGH[node]):
                    self.g[node] = -1
                    for nei in StateOne._GRAGH[node]:
                        if self.g[nei]>=1 or nei in visited or nei in alive:
                            continue
                        else:
                            if self.g[nei] == -1:
                                continue
                            stack.append(nei)
                            alive.add(nei)

    def clear_region(self):
        return sum(1 for i in self.g if i >= 0)

class Action:
    def __init__(self, p_id, next_node_id):
        # keep: (p_id, next_node_id)
        self.p_id = p_id
        self.next_node_id = next_node_id

    def __hash__(self):
        return hash((self.p_id, self.next_node_id))

    def __str__(self):
        return "Action({}, {})".format(str(self.p_id), str(self.next_node_id))

class ActionOne:
    def __init__(self, node_id, next_node_id):
        # state[node_id]-1, state[node_id]+1
        self.node_id = node_id
        self.next_node_id = next_node_id

    def __hash__(self):
        return hash((self.node_id, self.next_node_id))

    def __str__(self):
        return "ActionOne({}, {})".format(str(self.node_id, self.next_node_id))

class Agent:
    def __init__(self, alpha=1.0, epsilon=0.05, gamma=0.8, q_value=None):
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.discount = float(gamma)
        self.q_value = Counter() if q_value is None else q_value
        self.is_testing = False

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def observe_transition(self, state, action, next_state, delta_reward):
        self.update(state, action, next_state, delta_reward)

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

    def save_model(self, path):
        with open(path, 'wb') as outfile:
            pickle.dump({
                "q_value":self.q_value,
                "alpha":self.alpha,
                "epsilon":self.epsilon,
                "discount":self.discount,
            }, outfile)
    
    @staticmethod
    def load_model(path):
        with open(path, 'rb') as infile:
            data = pickle.load(infile)
            return Agent(
                alpha=data["alpha"],
                epsilon=data["epsilon"],
                gamma=data["discount"],
                q_value=data["q_value"]
            )

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
        StateOne.set_value(num_p, graph)
        self.display = Display(graph, mapping, map_type, k, w, 
                fix_r=int(conf["fix_r"]), fix_c=int(conf["fix_c"]), unit=int(conf["unit"]))
        self.state = StateOne()
        self.state_list = []

        # reward setting
        self.step_limit = 100
        self.step = 0
        self.prev = 0

    def get_current_state(self):
        return self.state

    def reset(self):
        self.state = StateOne()
        self.step = 0
        self.prev = 0
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
            return 150
        if self.step == self.step_limit:
            return -150
        
        clear_reg = self.state.clear_region()
        #if clear_reg < self.prev:
        #    penality = (self.prev - clear_reg) * 3
        #else:
        #    penality = 0

        return clear_reg - self.step*0.2 #- penality

    def get_possible_actions(self):
        return self.state.get_legal_action()

    def display_state(self, t=10):
        print("\rstep={}".format(self.step), end="   ")
        self.display.draw(self.state, t)

    def update_state_list(self, state):
        self.state_list.append(state)

    def do_action(self, action):
        self.state.do_action(action)
        self.step += 1
        return self.state, self.reward()

    def replay(self, t=20):
        for s in self.state_list:
            self.display.draw(s, t)

def run_episode(env, e, show, agent, discount):
    returns = 0
    totalDiscount = 1.0
    env.reset()

    while True:
        state = env.get_current_state().deep_copy()
        env.update_state_list(state)

        if show:
            #print(state)
            env.display_state(20)

        # check goal
        g1, g2 = env.is_goal()
        if g1:
            if g2:
                return returns, True
                print("   clean!!!!!!!!!", )
                #env.replay()
                #quit()
            return returns, False
        
        # make decision - choose action
        action = agent.get_action(state) 

        # do action
        next_state, reward = env.do_action(action)

        # update
        agent.observe_transition(state, action, next_state, reward)

        returns += reward * totalDiscount
        totalDiscount *= discount
        #print(state, reward, next_state, action)

def test_episode(env, show, agent, discount, t=20):
    returns = 0
    totalDiscount = 1.0
    env.reset()

    while True:
        state = env.get_current_state().deep_copy()
        env.update_state_list(state)
        
        # check goal
        g1, g2 = env.is_goal()
        if g1:
            if show:
                env.replay(t)
            if g2:
                return returns, True
                print("clean!!!!!!!!!")
            return returns, False

        action = agent.get_policy(state)
        next_state, reward = env.do_action(action)
        returns += reward * totalDiscount
        totalDiscount *= discount

def training():
    discount = 0.8
    """
    num_p = 2
    map_type = "tree"
    k = 1
    w = 1
    """
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
    num_p = 4
    map_type = "ladder"
    k = 2
    w = 2
    epsilon = 0.1
    learning_rate = 0.05
    agent = Agent(gamma=discount, epsilon=epsilon, alpha=learning_rate)
    env = Environment(num_p, map_type, k, w)
    array = Array(500, np.float32)
    model_dir = os.path.join("model", "{}_k{}_w{}".format(map_type, k, w))
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    goal_array = Array(500, np.float32)
    for e in range(1, 500001):
        returns, goal = run_episode(env, e, e%2000==0, agent, discount)
        array.append(returns)
        goal_array.append(float(goal))

        if e % 10000 == 0:
            epsilon *= 0.8
            epsilon = max(0.1, epsilon)
            agent.set_epsilon(epsilon)
            agent.save_model(os.path.join(model_dir, "model_e{}.rl".format(e)))
        
        if e % 50 == 0:
            print("\r e={}, epi={:.4f} returns={:.6f}, goal={:6f}, num={}".format(e, agent.epsilon, array.average(), goal_array.average(), len(agent.q_value)), end="       ")
        
        if e % 2000 == 0:
            returns, goal = test_episode(env, True, agent, discount)
            print()
            print("testing epoch:{}, returns:{:.6f}".format(e, returns))

def testing():
    # setting
    num_p = 3
    map_type = "tree" #"ladder"/"tree"
    k = 1
    w = 2
    discount = 0.8
    model_dir = os.path.join("model", "{}_k{}_w{}".format(map_type, k, w))
    testing_e = 15000

    # load model
    agent = Agent.load_model(os.path.join(model_dir, "model_e{}.rl".format(testing_e)))
    env = Environment(num_p, map_type, k, w)

    for i in range(0, 10):
        returns = test_episode(env, True, agent, discount, t=50)
        print("iteration: {}, returns: {}".format(i, returns))

if __name__ == "__main__":
    training()
    #testing()
