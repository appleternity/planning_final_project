class State:
    def __init__(self):
        # keep: purseur position, contaminated region
        pass
    
    def get_successors(self):
        # return [(next_state, action), (next_state, action), ...]
        pass
    
    def is_goal(self):
        # return True/False
        pass

    def do_action(self, action):
        # return next_state
        pass

    def deep_copy(self):
        # return a clone of self
        pass

    def reset(self):
        pass

class Action:
    def __init__(self):
        # keep: (p_id, next_node_id)
        pass

class Agent:
    def __init__(self):
        pass

    def get_qvalue(self, state, action):
        pass

    def compute_value_from_qvalue(self, state):
        pass

    def compute_action_from_qvalue(self, state):
        pass

    def get_action(self, state):
        pass

    def update(self, state, action , next_state, reward):
        pass

    def get_policy(self):
        pass

    def get_value(self):
        pass

class Environment:
    def __init__(self):
        # num_p, graph
        pass

    def get_current_state(self):
        pass

    def reset(self):
        pass

    def is_goal(self):
        pass

    def reward(self):
        pass

    def training(self):
        pass

def training():
    pass


def main():
    pass

if __name__ == "__main__":
    main()
