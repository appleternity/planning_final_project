from collections import defaultdict
from collections import deque
import heapq
import datetime
import json

def create_window_graph():

    #  0-0-0-0-0-0-0-0-0-0
    #  |     |     |     |
    #  0     0     0     0
    #  |     |     |     |
    #  0     0     0     0
    #  |     |     |     |
    #  0-0-0-0-0-0-0-0-0-0
    #  |     |     |     |
    #  0     0     0     0
    #  |     |     |     |
    #  0     0     0     0
    #  |     |     |     |
    #  0-0-0-0-0-0-0-0-0-0

    # create window graph with 45 vertex
    graph = defaultdict(list)


    graph[0] = [1,30]
    for i in range(1,9):
        graph[i] = [i-1,i+1]
    graph[3].append(31)
    graph[6].append(32)
    graph[9] = [8, 33]

    graph[10] = [13,34,38]
    for i in range(11, 19):
        graph[i] = [i - 1, i + 1]
    graph[13] += [35,39]
    graph[16] += [36,40]
    graph[19] = [16,37,41]

    graph[20] = [23,42]
    for i in range(21, 29):
        graph[i] = [i - 1, i + 1]
    graph[23] += [43]
    graph[26] += [44]
    graph[29]  = [26, 45]


    for i in range(30,34):
        graph[i] = [(i-30)*3, i+4]

    for i in range(34, 38):
        graph[i] = [(i - 4), (i-34)*3 + 10]

    for i in range(38,42):
        graph[i] = [(i - 38) * 3+10, i + 4]

    for i in range(42, 46):
        graph[i] = [i-4, (i - 42) * 3 + 20]


    #for i in graph:
    #    print(i, graph[i])

    return graph

def create_simple_graph():
    #
    #  0
    #  |
    #  0
    #  |
    #  0-0-0
    #  |
    #  0
    #  |
    #  0

    # create window graph with 45 vertex
    graph = defaultdict(list)
    graph[0] = [1]
    graph[1] = [0, 2]
    graph[2] = [1, 3, 5]
    graph[3] = [2, 4]
    graph[4] = [3]
    graph[5] = [2, 6]
    graph[6] = [5]

    # for i in graph:
    #     print(i, graph[i])

    return graph

def create_Ladder_graph():

    '''
    
    0-0-0-0-0
    | |   | |
    0-0   0-0
    | |   | |
    0-0-0-0-0    
    
    '''

    graph = defaultdict(list)
    graph[0] = [1, 10, 11]
    graph[1] = [0, 2, 10, 11]
    graph[2] = [1, 3]
    graph[3] = [2, 4, 12, 13]
    graph[4] = [3, 12, 13]
    graph[5] = [6, 10, 11]
    graph[6] = [5, 7, 10, 11]
    graph[7] = [6, 8]
    graph[8] = [7, 9, 12, 13]
    graph[9] = [8, 12, 13]
    graph[10] = [0, 5, 11]
    graph[11] = [1, 6, 10]
    graph[12] = [3, 8, 13]
    graph[13] = [4, 9, 12]

    return graph

def is_goal(state):

    # print('-----is_goal-----')
    # print(state)
    # state = (p, G)

    return sum(state[1]) == 0

def contaminate(p, ps, dirty):


    # print('=== contaminate===')
    # print(p, ps, dirty)
    # print('=== contaminate===')

    stack = [p]
    new_dirty = list(dirty[:])
    new_dirty[p] = 0

    # print(p, ps, new_dirty)
    visited = set()
    while stack:
        node = stack.pop()

        # print('here', node, [ new_dirty[nei] for nei in graph[node]])
        if any([ new_dirty[nei] for nei in graph[node]]):
            # print('nei exist dirty', node)
            new_dirty[node] = 1
            for nei in graph[node]:
                # print('nei = ', nei)
                if nei in ps or nei in visited:
                    continue
                else:
                    # print('**')
                    visited.add(nei)
                    if not dirty[nei]:
                        stack.append(nei)
                    new_dirty[nei] = 1

    # print('new_dirty = ', new_dirty)

    return new_dirty

def get_successors(state):

    # pass
    # state = (pursuers , G)

    pursuers, dirty =  state
    pursuers = list(pursuers)
    # print orinal p
    # print('original = ', pursuers)
    next_state = set()
    next_pursurers = set()

    for i in range(len(pursuers)):
        for p in graph[pursuers[i]]:
            next_p = tuple(pursuers[:i] + [p] + pursuers[i+1:])
            if next_p not in next_pursurers:
                next_pursurers.add(next_p)


    for next_ps in next_pursurers:
        # print('next_ps = ',next_ps)
        for origin_p, next_p in zip(pursuers, next_ps):
            if origin_p != next_p:
                # print('origin_p, next_p = ', pursuers, next_ps, dirty)
                if origin_p not in next_ps:
                    new_dirty = list(dirty[:])
                    new_dirty[next_p] = 0
                    new_dirty = contaminate(origin_p, next_ps, new_dirty)
                    # new_dirty[next_p] = 0
                    next_state.add((tuple(next_ps), tuple(new_dirty)))

                else:
                    new_dirty = list(dirty[:])
                    new_dirty[next_p] = 0
                    # print(new_dirty)
                    next_state.add((tuple(next_ps), tuple(new_dirty)))

    # print('next_state = ')
    # for i in next_state:
    #     print(i)

    # return a list of (p, dirty)
    return next_state


def not_bfs(state):

    # state= pursuers, tuple(dirty)
    queue = [] # deque([])
    visited = set()
    heapq.heappush(queue, (-sum(dirty),(tuple(state), [pursuers])))
    # queue.append((tuple(state), [pursuers]))
    counter = 0
    while queue:
        temp = heapq.heappop(queue)
        _, (node, cur_path) =  temp # queue.popleft()
        # print('node, cur_path = ', node, cur_path)# , visited)
        if node not in visited:
            visited.add(node)
            # print('len(visited) = ', len(visited))
            if counter %10000 == 0:
                print(counter)

            counter +=1
            if is_goal(node):
                # print(node)
                print('GOAL!!!!!')
                print('counter = ', counter)
                return cur_path
            for suc in get_successors(node):
                # print('----')
                # print('suc = ', suc)
                heapq.heappush(queue, (-sum(dirty), (suc, cur_path + [suc[0]])))
                #queue.append((suc, cur_path + [suc[0]]))

    # print('fuck you no solution')
    return []

def bfs(state):

    # state= pursuers, tuple(dirty)
    queue = deque([])
    visited = set()
    queue.append((tuple(state), [pursuers]))
    counter = 0
    while queue:
        node, cur_path = queue.popleft()
        # print('node, cur_path = ', node, cur_path)# , visited)
        if node not in visited:
            visited.add(node)
            # print('len(visited) = ', len(visited))
            if counter %10000 == 0:
                print(counter, 'node = ',node[0], len(cur_path))
            counter +=1
            if is_goal(node):
                print(node)
                print('GOAL!!!!!')
                print('counter = ', counter)
                return cur_path
            for suc in get_successors(node):
                # print('----')
                # print('suc = ', suc)
                queue.append((suc, cur_path + [suc[0]]))

    # print('fuck you no solution')
    return []

def dfs(state):

    # state= pursuers, tuple(dirty)
    queue = deque([])
    visited = set()
    queue.append((tuple(state), [pursuers]))
    counter = 0
    while queue:
        node, cur_path = queue.pop()
        # print('node, cur_path = ', node, cur_path)# , visited)
        if node not in visited:
            visited.add(node)
            # print('len(visited) = ', len(visited))
            if counter %10000 == 0:
                print(counter, 'node = ',node[0], len(cur_path))
            counter +=1
            if is_goal(node):
                print(node)
                print('GOAL!!!!!')
                print('counter = ', counter)
                return cur_path
            for suc in get_successors(node):
                # print('----')
                # print('suc = ', suc)
                queue.append((suc, cur_path + [suc[0]]))

    # print('fuck you no solution')
    return []
# graph = create_simple_graph()
# graph = create_window_graph()
graph = create_Ladder_graph()

# filename = "./ladder_k4_w1_state.json"
# with open(filename, 'r') as infile:
#     graph = json.load(infile)
#     graph = {
#        int(k):[int(vv) for vv in v]
#        for k, v in graph.items()
#    }

print(graph)

t1 = datetime.datetime.now()
for i in range(1,10):
    pursuers = tuple(0 for i in range(i))
    dirty = [1] * len(graph)
    dirty[0] = 0
    # dirty[1] = 0
    # dirty[3] = 0
    state = (pursuers, tuple(dirty))

    # options of dfs, bfs, not_bfs
    ans = dfs(state)

    if ans:
        # print(i, ans)
        # for _ in ans:
        #     print(_)
        print('path len = ', len(ans))
        break

    else:
        print('no ans for ', i)

t2 = datetime.datetime.now()

print((t2-t1).total_seconds())
