from collections import defaultdict
from collections import deque
import heapq
import datetime
import json
import os
import sys
def create_window_graph():

    # create window graph with 45 vertex
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


    graph = defaultdict(list)

    graph[0] = [1,30]
    for i in range(1,9):
        graph[i] = [i-1,i+1]
    graph[3].append(31)
    graph[6].append(32)
    graph[9] = [8, 33]

    graph[10] = [11,34,38] # fixed 13-> 11
    for i in range(11, 19):
        graph[i] = [i - 1, i + 1]
    graph[13] += [35,39]
    graph[16] += [36,40]
    graph[19] = [18,37,41] # fixed 16-> 18

    graph[20] = [21,42]
    for i in range(21, 29):
        graph[i] = [i - 1, i + 1]
    graph[23] += [43]
    graph[26] += [44]
    graph[29]  = [28, 45] # fixed 26-> 28


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

def create_simple_h_graph():
    #
    #  0
    #  |
    #  0   0
    #  |   |
    #  0-0-0
    #  |   |
    #  0   0
    #  |
    #  0

    # create simple graph with 9 vertex
    graph = defaultdict(list)
    graph[0] = [1]
    graph[1] = [0, 2]
    graph[2] = [1, 3, 5]
    graph[3] = [2, 4]
    graph[4] = [3]
    graph[5] = [2, 6]
    graph[6] = [5, 7]
    graph[7] = [6]
    graph[8] = [7]

    # for i in graph:
    #     print(i, graph[i])

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

    # create simgle graph with 7 vertex
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
    |x|   |x|
    0-0   0-0
    |x|   |x|
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
    graph[10] = [0, 1, 5, 6, 11]
    graph[11] = [0, 1, 5, 6, 10]
    graph[12] = [3, 4, 8, 9, 13]
    graph[13] = [3, 4, 8, 9, 12]

    return graph

def is_goal(state):

    # print('-----is_goal-----')
    # print(state)
    # state = (p, G)

    return sum(state[1]) == 0

def contaminate(p, ps, dirty):

    # contaminate after move
    # p:  original pos for a pursuer
    # ps: list of pos ofr pursuers
    # dirty: contaminate state

    # print('=== contaminate===')
    # print(p, ps, dirty)
    # print('=== contaminate===')

    stack = [p]
    # new_dirty = dirty[:p] + (0,) + dirty[p + 1:]
    new_dirty = list(dirty[:])
    new_dirty[p] = 0
    visited = set()
    while stack:
        node = stack.pop()

        # contaminate node if any neighbor is contaminated
        if any([ new_dirty[nei] for nei in graph[node]]):
            # new_dirty = new_dirty[:node] + (1,) + new_dirty[node + 1:]
            new_dirty[node] = 1
            for nei in graph[node]:
                if nei in ps or nei in visited:
                    continue
                else:
                    visited.add(nei)
                    if not dirty[nei]:
                        stack.append(nei)
                    new_dirty[nei] = 1

    return tuple(new_dirty)

def get_successors(state):

    # state = (pursuers , G)

    pursuers, dirty =  state
    next_state = set()
    next_pursurers = set()

    for i in range(len(pursuers)):
        for p in graph[pursuers[i]]:
            next_p = pursuers[:i] + (p,) + pursuers[i+1:]
            if next_p not in next_pursurers:
                next_pursurers.add(next_p)


    for next_ps in next_pursurers:
        for origin_p, next_p in zip(pursuers, next_ps):
            if origin_p != next_p:
                new_dirty = dirty[:next_p] + (0,) + dirty[next_p + 1:]
                if origin_p not in next_ps:

                    new_dirty = contaminate(origin_p, next_ps, new_dirty)

                next_state.add((next_ps, new_dirty))

    # print('next_state = ')
    # for i in next_state:
    #     print(i)

    # return a list of (p, dirty)
    return next_state

def not_bfs(state):

    # best first search

    # state= pursuers, tuple(dirty)
    queue = []
    visited = set()
    heapq.heappush(queue, (sum(dirty),(state, [state])))
    # queue.append((tuple(state), [pursuers]))
    counter = 0
    while queue:
        temp = heapq.heappop(queue)
        _, (node, cur_path) =  temp # queue.popleft()
        # print(queue[:3])
        # print('node, cur_path = ', node, cur_path)# , visited)
        if node not in visited:
            visited.add(node)
            # print('len(visited) = ', len(visited))
            if counter and counter %10000 == 0:
                print(counter)

            counter +=1
            if is_goal(node):
                # print(node)
                print('GOAL!!!!!')
                # print('counter = ', counter)
                return counter, cur_path
            for suc in get_successors(node):
                heapq.heappush(queue, (sum(suc[1]), (suc, cur_path + [suc])))
                #queue.append((suc, cur_path + [suc[0]]))

    return counter, []

def bfs(state):

    # state= pursuers, tuple(dirty)
    print('---- searching with bfs, N=', len(state[0]))
    queue = deque([])
    visited = set()
    queue.append((state, [state]))
    counter = 0
    while queue:
        node, cur_path = queue.popleft()
        # print('node, cur_path = ', node, cur_path)# , visited)
        if node not in visited:
            visited.add(node)
            # print('len(visited) = ', len(visited))
            if counter and counter % 10000 == 0:
                print('iteration:', counter)# , 'node = ',node[0], len(cur_path))
            counter +=1
            if is_goal(node):
                # print(node)
                print('GOAL!!!!!')
                print('counter = ', counter)
                return counter, cur_path
            for suc in get_successors(node):
                # print('----')
                # print('suc = ', suc)
                queue.append((suc, cur_path + [suc]))

    # print('fuck you no solution')
    return counter, []

def dfs(state):

    # state= pursuers, tuple(dirty)
    queue = deque([])
    visited = set()
    queue.append((state, [state]))
    counter = 0
    while queue:
        node, cur_path = queue.pop()
        print('node, cur_path = ', node, cur_path)# , visited)
        if node not in visited:
            visited.add(node)
            # print('len(visited) = ', len(visited))
            if counter %10000 == 0:
                print(counter, 'node = ',node[0], len(cur_path))
            counter +=1
            if is_goal(node):
                # print(node)
                print('GOAL!!!!!')
                print('counter = ', counter)
                return counter, cur_path
            for suc in get_successors(node):
                # print('----')
                # print('suc = ', suc)
                queue.append((suc, cur_path + [suc]))

    # print('fuck you no solution')
    return counter, []

# graph = create_simple_graph()
# graph = create_window_graph()
# graph = create_Ladder_graph()
#
# filename = "../ladder_k4_w1_state.json"


def read_json(filename):
    # parse json file
    with open(filename, 'r') as infile:
        graph = json.load(infile)
        graph = {
           int(k):[int(vv) for vv in v]
           for k, v in graph.items()
       }
    return graph


def sort_func(x):
    # sort json file with number of vertex

    # print('-----sorted--------')
    # print(x, len(read_json('./state/'+x)))
    # if x[0] != 'w':
    #     x = x.replace("_", '', 1).split('_')
    # else:
    #     x = x.split('_')
    # print(x)
    return len(read_json('./state/'+x))
    # return (x[1],x[2],x[0])


# read files
path = './'
files = []
# for filename in os.listdir(path):
#     if filename != '.DS_Store':
#         files.append(filename)
output_s = ''
if len(sys.argv) == 2:
    file = sys.argv[1]
    # print(sys.argv)

    # files = sorted(files, key=sort_func)
    # for file in files:

    print('--------', file, '------------')
    o_file = file.replace('json', 'txt')
    o_file =  o_file


    # if file !=  '_ladder_k2_w2_state.json':
    #     continue

    output_s += str(file) + '\n'

    graph = read_json(path +file)
else:
    graph = create_simple_graph()
    o_file = 'simple_graph.txt'
output_s += 'Number of vertices in graph: ' + str(len(graph)) + '\n'

t1 = datetime.datetime.now()


# search from N=1 until find answer
for i in range(1, 10):
    pursuers = tuple(0 for i in range(i))
    dirty = (0,) + tuple(1 for _ in range(len(graph) - 1))
    state = (pursuers, dirty)

    # options of dfs, bfs, not_bfs
    iterations, ans = bfs(state)

    if ans:
        # print('Ans for N = ', i)
        output_s += str('Minimum pursuers is N = ' +  str(i) + '\n')
        output_s = output_s + "Shortest path start: \n"
        for _ in ans:
            #     print(_)

            output_s = output_s + "\t" + str(_[0]) + '\n'
        print('\npath len = ', len(ans))
        # print('counter = ', iterations)

        output_s += 'Path length is ' +  str(len(ans)) + '\n'
        # output_s += 'Counter = ' + str(iterations) + '\n'
        break

    else:
        pass
        # print('No ans for N = ', i )
        #output_s += str('No ans for N = ' +  str(i) + '\n')

t2 = datetime.datetime.now()
output_s += 'Time elapsed: '
output_s += str((t2-t1).total_seconds()) +' s\n'

print(o_file)
with open('./ans_brute_force'  + o_file[6:]  ,  'w+') as f:
    f.write(output_s)