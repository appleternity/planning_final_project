from collections import defaultdict

def create_graph():

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

    for i in range(42, 45):
        graph[i] = [i-4, (i - 42) * 3 + 20]


    for i in graph:
        print(i, graph[i])

create_graph()