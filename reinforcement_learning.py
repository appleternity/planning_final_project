from copy import copy, deepcopy
import random
from util import Counter, Array, flipcoin, parsing_config, choice_with_distribution
from display import Display
import os, os.path
import json
import numpy as np
import pickle
from pprint import pprint
#from brute_force import create_window_graph
import imageio
from argparse import ArgumentParser

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
    
    def __init__(self, g=None, keep=None, p=None):
        self.g = g if g is not None else [StateOne._N if i == 0 else -1 for i, _ in enumerate(StateOne._GRAGH)]
        self.keep = keep
        if self.keep:
            self.pursuers = tuple(0 for _ in range(StateOne._N)) if p is None else p

    def __hash__(self):
        return hash(tuple(self.g))

    def __str__(self):
        return "StateOne({})".format(str(self.g))
    
    def is_goal(self):
        return sum(1 for g in self.g if g == -1) == 0

    def deep_copy(self):
        if self.keep:
            return StateOne(list(self.g), keep=True, p=self.pursuers)
        else:
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

        if self.keep:
            ps = list(self.pursuers)
            ps[ps.index(action.node_id)] = action.next_node_id
            self.pursuers = tuple(ps)
        
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

    def boundary_pursuers(self):
        num = 0
        for i, p in enumerate(self.g):
            if p >= 1:
                nei_state = [self.g[nei] for nei in StateOne._GRAGH[i]]
                if 0 in nei_state and -1 in nei_state:
                    num += p
        return num

    def distance_boundary_pursuers(self):
        pur = []
        border = [0] * len(self.g)
        b_exist = False
        for idx, g in enumerate(self.g):
            if g >= 1:
                for i in range(g):
                    pur.append((idx, 0))

                nei_state = [self.g[nei] for nei in StateOne._GRAGH[idx]]
                if 0 in nei_state and -1 in nei_state:
                    border[idx] = 1
                    b_exist = True
                else:
                    border[idx] = 0

        total_dis = 0
        if b_exist:
            for p in pur:
                if border[p[0]] == 1:
                    continue
                stack = [p]
                visited = set()
                while stack:
                    (node,dis) = stack.pop(0)
                    if border[node]:
                        total_dis+= dis
                        break
                    for nei in StateOne._GRAGH[node]:
                        if  nei in visited:
                            continue
                        else:
                            visited.add(nei)
                            stack.append((nei,dis+1))

        return total_dis

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
    def __init__(self, alpha=1.0, epsilon=0.05, gamma=0.8, q_value=None, arg=None):
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.discount = float(gamma)
        self.q_value = Counter() if q_value is None else q_value
        self.is_testing = False
        self.counter = Counter()
        self.arg = arg

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
            action = self.random_action(state, legal_actions)
        else:
            action = self.compute_action_from_qvalue(state)

        return action

    def random_action(self, state, legal_actions):
        if self.arg.exploration == "boltzmann":
            # semi-uniform distributed
            p = np.array([self.get_qvalue(state, a) for a in legal_actions])
            p = np.exp(p) + 1
            p = p / np.sum(p)
            #p = p / np.sum(p)
            return choice_with_distribution(legal_actions, p)
    
        if self.arg.exploration == "counter":
            # counter_based
            f_a = np.array([self.get_qvalue(state, a) for a in legal_actions])
            e_c = np.array([1/(self.counter[(hash(state), hash(a))]+1) for a in legal_actions])
            p = f_a * 0.5 + e_c*10
            p = p / np.sum(p)
            select_a = choice_with_distribution(legal_actions, p)
            self.counter[(hash(state), hash(select_a))] += 1
            return select_a

        if self.arg.exploration == "random":
            # pure random
            return random.choice(legal_actions)

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
    def load_model(path, arg):
        with open(path, 'rb') as infile:
            data = pickle.load(infile)
            return Agent(
                alpha=data["alpha"],
                epsilon=data["epsilon"],
                gamma=data["discount"],
                q_value=data["q_value"],
                arg=arg
            )

class Environment:
    def __init__(self, num_p, map_type, k, w, keep_pursuer=False, arg=None):
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
        self.state = StateOne(keep=keep_pursuer)
        self.state_list = []
        self.keep_pursuer = keep_pursuer
        self.arg = arg

        # reward setting
        self.step_limit = 100
        self.step = 0
        self.prev = 0

    def get_current_state(self):
        return self.state

    def reset(self):
        self.state = StateOne(keep=self.keep_pursuer)
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
        # version testing: dynamic reward + boundary (best)
        """
        clear_reg = self.state.clear_region()
        if self.state.is_goal():
            return clear_reg*20
        if self.step == self.step_limit:
            b = self.state.boundary_pursuers()
            bd = self.state.distance_boundary_pursuers()
            return -1000+clear_reg*10+b*100-bd*5
        return clear_reg - self.step*0.3
        """
        if self.arg.reward_function == "boundary":
            # version 1: dynamic reward + boundary (best)
            clear_reg = self.state.clear_region()
            if self.state.is_goal():
                return clear_reg*20
            if self.step == self.step_limit:
                b = self.state.boundary_pursuers()
                return -1000+clear_reg*5+b*100
            return clear_reg - self.step*0.3

        if self.arg.reward_function == "dynamic":
            # version 2: dynamic reward 
            clear_reg = self.state.clear_region()
            if self.state.is_goal():
                return clear_reg*20
            if self.step == self.step_limit:
                return -1000+clear_reg*10
            return clear_reg - self.step*0.3
            
        if self.arg.reward_function == "fix":
            # version 3: fix reward
            clear_reg = self.state.clear_region()
            if self.state.is_goal():
                return 150
            if self.step == self.step_limit:
                return -150
            return clear_reg - self.step*0.3

    def get_possible_actions(self):
        return self.state.get_legal_action()

    def display_state(self, t=10):
        #print("\rstep={}".format(self.step), end="   ")
        self.display.draw(self.state, t)

    def update_state_list(self, state):
        self.state_list.append(state)

    def do_action(self, action):
        self.state.do_action(action)
        self.step += 1
        return self.state, self.reward()

    def replay(self, t=20, save_dir=None):
        if save_dir:
            images = []
        for i, s in enumerate(self.state_list):
            if save_dir:
                img = self.display.draw(s, t, os.path.join(save_dir, "image_{:0>3}.png".format(i)))
                new_img = np.asarray(img)
                new_img[:,:,:] = img[:,:,::-1]
                images.append(img)
            else:
                self.display.draw(s, t)
        if save_dir:
            imageio.mimsave(os.path.join(save_dir, "demo.gif"), images, duration=0.1)

def run_episode(env, e, show, agent, discount):
    discount = 0.95
    returns = 0
    totalDiscount = 1.0
    env.reset()
    #reward_list = []

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
        #reward_list.append(reward)

        # update
        agent.observe_transition(state, action, next_state, reward)

        returns += reward * totalDiscount
        totalDiscount *= discount
        #print(state, reward, next_state, action)

def test_episode(env, show, agent, discount, t=20, save_dir=None):
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
                env.replay(t, save_dir)
            if g2:
                return returns, True
                print("clean!!!!!!!!!")
            return returns, False

        action = agent.get_policy(state)
        next_state, reward = env.do_action(action)
        returns += reward * totalDiscount
        totalDiscount *= discount

def training(arg):
    discount = 0.8
    """
    ladder_k2_w2 => step_limit 120, epsilon 0.3
    """
    num_p = arg.n
    map_type = arg.map
    k = arg.k
    w = arg.w
    epsilon = 0.3
    learning_rate = 0.01
    agent = Agent(gamma=discount, epsilon=epsilon, alpha=learning_rate, arg=arg)
    env = Environment(num_p, map_type, k, w, arg=arg)
    array = Array(500, np.float32)
    model_dir = os.path.join("model", "{}_k{}_w{}".format(map_type, k, w))
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    log_filename = os.path.join("log", "{}_k{}_w{}".format(map_type, k, w))
    log_file = open(log_filename, 'w', encoding='utf-8')

    goal_array = Array(500, np.float32)
    for e in range(1, 500001):
        returns, goal = run_episode(env, e, e%2000==0, agent, discount)
        #returns, goal = run_episode(env, e, False, agent, discount)
        array.append(returns)
        goal_array.append(float(goal))
        
        # epsilon decay
        if e % 10000 == 0 and arg.exploration == "random":
            epsilon *= 0.8
            epsilon = max(0.05, epsilon)
            agent.set_epsilon(epsilon)

        # save model
        if e % 5000 == 0:
            agent.save_model(os.path.join(model_dir, "model_e{}.rl".format(e)))
        
        # print information
        if e % 50 == 0:
            print("\r e={}, epi={:.4f} returns={:.6f}, goal={:6f}, num={}".format(e, agent.epsilon, array.average(), goal_array.average(), len(agent.q_value)), end="       ")
            log_file.write(json.dumps({"e":e, "epi":agent.epsilon, "returns":float(array.average()), "goal":float(goal_array.average()), "num_state":len(agent.q_value)})+"\n") 
            log_file.flush()

        # run testing
        if e % 2000 == 0:
            returns, goal = test_episode(env, True, agent, discount)
            #returns, goal = test_episode(env, False, agent, discount)
            print()
            print("testing epoch:{}, returns:{:.6f}".format(e, returns))

def testing(arg):
    # setting
    num_p = arg.n
    map_type = arg.map
    k = arg.k
    w = arg.w
    discount = 0.8
    model_dir = os.path.join("model", "{}_k{}_w{}".format(map_type, k, w))
    if arg.testing_episode is None:
        filenames = os.listdir(model_dir)
        filename = sorted(filenames)[-1]
        testing_e = int(filename[7:-3])
        print("Didn't specify the testing episode, running on model_e{}.rl".format(testing_e))
    else:
        testing_e = arg.testing_episode

    # load model
    agent = Agent.load_model(os.path.join(model_dir, "model_e{}.rl".format(testing_e)), arg=arg)
    env = Environment(num_p, map_type, k, w, keep_pursuer=True, arg=arg)

    for i in range(0, 10):
        if i == 0:
            if arg.demo:
                demo_dir = os.path.join("demo", "{}_k{}_w{}_e{}".format(map_type, k, w, testing_e))
                if not os.path.isdir(demo_dir):
                    os.mkdir(demo_dir)
                returns = test_episode(env, True, agent, discount, t=40, save_dir=demo_dir)
            else:
                returns = test_episode(env, True, agent, discount, t=40)
        else:
            returns = test_episode(env, True, agent, discount, t=40)
        print("iteration: {}, returns: {}".format(i, returns))

def parse_arg():
    parser = ArgumentParser()
    parser.add_argument("-m", "--map", dest="map", type=str, help="map type [ladder, tree]", default="ladder")
    parser.add_argument("-k", "--k", dest="k", type=int, help="branching factor [Integer]", default=2)
    parser.add_argument("-w", "--w", dest="w", type=int, help="path width [Integer]", default=1)
    parser.add_argument("-n", "--n", dest="n", type=int, help="number of pursuers [Integer]", default=2)
    parser.add_argument("-p", "--phase", dest="phase", type=str, help="phase [training, testing]", default="training")
    parser.add_argument("-e", "--testing_episode", dest="testing_episode", type=int, help="testing episode [Integer]", default=None)
    parser.add_argument("-d", "--save_demo", dest="demo", action="store_true", help="output demo file or not", default=False)
    
    parser.add_argument("-r", "--reward", dest="reward_function", type=str, help="reward function [fix, dynamic, boundary]", default="boundary")
    parser.add_argument("-x", "--explore", dest="exploration", type=str, help="exploration strategy [random, boltzmann, counter]", default="boltzmann")
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    if not os.path.isdir("model"):
        os.mkdir("model")
    if not os.path.isdir("log"):
        os.mkdir("log")
    if not os.path.isdir("demo"):
        os.mkdir("demo")

    arg = parse_arg()

    if arg.phase == "training":
        training(arg)
    elif arg.phase == "testing":
        testing(arg)
    else:
        print("Please enter a valid phase (training/testing)")
