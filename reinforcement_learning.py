from copy import copy, deepcopy
import random


class Counter(dict):
    """
    A counter keeps track of counts for a set of keys.

    The counter class is an extension of the standard python
    dictionary type.  It is specialized to have number values
    (integers or floats), and includes a handful of additional
    functions to ease the task of counting data.  In particular,
    all keys are defaulted to have value 0.  Using a dictionary:

    a = {}
    print a['test']

    would give an error, while the Counter class analogue:

    >>> a = Counter()
    >>> print a['test']
    0

    returns the default 0 value. Note that to reference a key
    that you know is contained in the counter,
    you can still use the dictionary syntax:

    >>> a = Counter()
    >>> a['test'] = 2
    >>> print a['test']
    2

    This is very useful for counting things without initializing their counts,
    see for example:

    >>> a['blah'] += 1
    >>> print a['blah']
    1

    The counter also includes additional functionality useful in implementing
    the classifiers for this assignment.  Two counters can be added,
    subtracted or multiplied together.  See below for details.  They can
    also be normalized and their total count and arg max can be extracted.
    """
    def __getitem__(self, idx):
        self.setdefault(idx, 0)
        return dict.__getitem__(self, idx)

    def incrementAll(self, keys, count):
        """
        Increments all elements of keys by the same count.

        >>> a = Counter()
        >>> a.incrementAll(['one','two', 'three'], 1)
        >>> a['one']
        1
        >>> a['two']
        1
        """
        for key in keys:
            self[key] += count

    def argMax(self):
        """
        Returns the key with the highest value.
        """
        if len(self.keys()) == 0: return None
        all = self.items()
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def sortedKeys(self):
        """
        Returns a list of keys sorted by their values.  Keys
        with the highest values will appear first.

        >>> a = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> a['third'] = 1
        >>> a.sortedKeys()
        ['second', 'third', 'first']
        """
        sortedItems = self.items()
        sign = lambda x: (1, -1)[x < 0]
        compare = lambda x, y:  sign(y[1] - x[1])
        sortedItems.sort(cmp=compare)
        return [x[0] for x in sortedItems]

    def totalCount(self):
        """
        Returns the sum of counts for all keys.
        """
        return sum(self.values())

    def normalize(self):
        """
        Edits the counter such that the total count of all
        keys sums to 1.  The ratio of counts for all keys
        will remain the same. Note that normalizing an empty
        Counter will result in an error.
        """
        total = float(self.totalCount())
        if total == 0: return
        for key in self.keys():
            self[key] = self[key] / total

    def divideAll(self, divisor):
        """
        Divides all counts by divisor
        """
        divisor = float(divisor)
        for key in self:
            self[key] /= divisor

    def copy(self):
        """
        Returns a copy of the counter
        """
        return Counter(dict.copy(self))

    def __mul__(self, y ):
        """
        Multiplying two counters gives the dot product of their vectors where
        each unique label is a vector element.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['second'] = 5
        >>> a['third'] = 1.5
        >>> a['fourth'] = 2.5
        >>> a * b
        14
        """
        sum = 0
        x = self
        if len(x) > len(y):
            x,y = y,x
        for key in x:
            if key not in y:
                continue
            sum += x[key] * y[key]
        return sum

    def __radd__(self, y):
        """
        Adding another counter to a counter increments the current counter
        by the values stored in the second counter.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['third'] = 1
        >>> a += b
        >>> a['first']
        1
        """
        for key, value in y.items():
            self[key] += value

    def __add__( self, y ):
        """
        Adding two counters gives a counter with the union of all keys and
        counts of the second added to counts of the first.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['third'] = 1
        >>> (a + b)['first']
        1
        """
        addend = Counter()
        for key in self:
            if key in y:
                addend[key] = self[key] + y[key]
            else:
                addend[key] = self[key]
        for key in y:
            if key in self:
                continue
            addend[key] = y[key]
        return addend

    def __sub__( self, y ):
        """
        Subtracting a counter from another gives a counter with the union of all keys and
        counts of the second subtracted from counts of the first.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['third'] = 1
        >>> (a - b)['first']
        -5
        """
        addend = Counter()
        for key in self:
            if key in y:
                addend[key] = self[key] - y[key]
            else:
                addend[key] = self[key]
        for key in y:
            if key in self:
                continue
            addend[key] = -1 * y[key]
        return addend

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


    def contaminate(self, p, ps, dirty):
        """
        return: tuple(new_dirty)
        """

        new_dirty = list(dirty[:])
        move_pos = None
        for p1, p2 in zip(p, ps):
            if p1 != p2:
                move_pos = p1
                new_dirty[p2] = 0
                new_dirty[p1] = 0
                break

        stack = [move_pos]
        visited = set()

        while stack:
            node = stack.pop()

            # contaminate node if any neighbor is contaminated
            if any([new_dirty[nei] for nei in State._GRAGH[node]]):
                # new_dirty = new_dirty[:node] + (1,) + new_dirty[node + 1:]
                new_dirty[node] = 1
                for nei in State._GRAGH[node]:
                    if nei in ps or nei in visited:
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
        next_pursuers = set()

        for i in range(len(pursuers)):
            for p in State._GRAGH[pursuers[i]]:
                next_p = (pursuers[:i] + (p,) + pursuers[i + 1:], Action(i, p))

                if next_p not in next_pursuers:
                    next_pursuers.add(next_p)

        return [i for i in next_pursuers]

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

        pass
    
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

        next_ps = self.pursuers[:action.p_id] + (action.next_node_id,) + self.pursuers[action.p_id + 1:]
        new_dirty = self.dirty[:action.p_id] + (0,) + self.dirty[action.p_id + 1:]

        if action.p_id not in next_ps:
            new_dirty = self.contaminate(action.p_id, next_ps, new_dirty)

        self.pursuers = next_ps
        self.dirty = self.contaminate(self.pursuers, next_ps, new_dirty)


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


class Agent:

    def __init__(self, alpha=1.0, epsilon=0.05, gamma=0.8, numTraining = 10):
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.discount = float(gamma)
        self.numTraining = int(numTraining)
        self.q_value = Counter()


    def get_qvalue(self, state, action):

        if (state, action) in self.q_value:
            return self.q_value[(state, action)]

        self.q_value[(state, action)] = 0.0
        return 0.0

    def compute_value_from_qvalue(self, state):

        if state.get_legal_action():
            values = Counter()
            for action in state.get_legal_action():
                values[action] = self.get_qvalue(state, action)

            return values[values.argMax()]

        return 0.0


    def compute_action_from_qvalue(self, state):

        moves = state.get_legal_action()
        if moves:
            values = Counter()
            for action in moves:
                values[action] = self.get_qvalue(state, action)
            return values.argMax()

        return None


    def get_action(self, state):

        def flipcoin(p):
            r = random.random()
            return r < p

        legal_actions = state.get_legal_action()
        action = None

        "*** YOUR CODE HERE ***"
        if not legal_actions :
            return action

        if flipcoin(self.epsilon):
            action = random.choice(legal_actions)
        else:
            action = self.compute_action_from_qvalue(state)

        return action


    def update(self, state, action , next_state, reward):
        self.q_value[(state, action)] = self.q_value[(state, action)] + self.alpha * (
                reward + self.discount * self.get_value(next_state) - self.get_qvalue(state, action))

    def get_policy(self, state):
        return self.compute_action_from_qvalue(state)

    def get_value(self, state):
        return self.compute_action_from_qvalue(state)

class Environment:
    def __init__(self):
        # num_p, graph
        # A = State(n, graph)

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
