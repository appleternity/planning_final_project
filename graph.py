from collections import defaultdict
import json

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
    graph[10] = [0, 1, 5, 6, 11]
    graph[11] = [0, 1, 5, 6, 10]
    graph[12] = [3, 4, 8, 9, 13]
    graph[13] = [3, 4, 8, 9, 12]

    return graph

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
    #  0   0
    #  |   |
    #  0   0
    #  |   |
    #  0-0-0
    #  |   |
    #  0   0
    #  |   |
    #  0   0

    # create window graph with 45 vertex
    graph = defaultdict(list)
    graph[0] = [1]
    graph[1] = [0, 2]
    graph[2] = [1, 3, 5]
    graph[3] = [2, 4]
    graph[4] = [3]
    graph[5] = [2, 6]
    graph[6] = [5,7, 9]
    graph[7] = [6, 8]
    graph[8] = [7]
    graph[9] = [6, 10]
    graph[10]= [9]

    # for i in graph:
    #     print(i, graph[i])

    return graph

def load_graph(filepath):
    with open(filepath, 'r') as infile:
        data = json.load(infile)

    data = {
        int(key):val
        for key, val in data.items()        
    }
    return data


