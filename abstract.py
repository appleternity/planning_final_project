from graph import *
from itertools import permutations
from pprint import pprint
import json
from datetime import datetime
import pickle
from collections import deque

count_repeat = 0
count_conflict = 0
count_cc = 0

def abstract(graph, num_p, version=""):
    t1 = datetime.now() 
    prev_time = datetime.now()

    S = {}
    abs_index = 0
    M = {}
    transition_table = {}
    cc_table = {}
    gamma = {}
    # TODO:put cc here
    Q = [tuple(0 for i in range(0, num_p))]
    visited = set()
    alive = set(Q)
    count = 0
    merge_history = []

    while Q:
        p = Q.pop()

        #alive.remove(p)
        visited.add(p)
        gamma[p] = None
        adj_s = set()
        count += 1
        if count % 100000 == 0:
            now = datetime.now()
            print(count, len(Q), (now-prev_time).total_seconds())
            prev_time = now

        # find children
        children = {
            tuple(ppp if i != ii else next_p for ii, ppp in enumerate(p))
            for i, pp in enumerate(p)
                for next_p in graph[pp]
        }

        for next_p in children:
            if next_p in visited:
                if gamma[p] == gamma[next_p]:
                    continue
                
                if cc_similiar(graph, p, next_p, transition_table, cc_table):
                    if gamma[p] is None:
                        gamma[p] = gamma[next_p]
                        #S[gamma[p]].append(p)
                        S[gamma[p]].add(p)
                    else:
                        # resolve conflict
                        abs1 = gamma[p]
                        abs2 = gamma[next_p]
                        
                        # merge S
                        S[abs1].update(S[abs2])
                        for temp_p in S[abs2]:
                            gamma[temp_p] = abs1
                        S.pop(abs2)

                        # TODO:merge M
                        merge_history.append((abs1, abs2))

                        """
                        M[abs1].update(M[abs2])
                        if abs1 in M[abs1]:
                            print("GG")
                            M[abs1].remove(abs1)
                        if abs2 in M[abs1]:
                            print("GG")
                            M[abs1].remove(abs2)
                        M.pop(abs2)
                        """
                else:
                    # Adjacent_s.insert(gamma_p')
                    #print(p, next_p)
                    adj_s.add(gamma[next_p])
            else:
                # Q.insert(p'), mark p' as alive
                if next_p not in alive:
                    Q.append(next_p)
                    alive.add(next_p)

        if gamma[p] is None:
            # S.insert(abstract(p)), gamma_p <-abstract(p)
            abs_p = abs_index
            abs_index += 1
            S[abs_p] = {p}
            gamma[p] = abs_p
            M[gamma[p]] = set()

        for a in adj_s:
            # update M(gamma_p, a)
            #if gamma[p] not in M:
            #    M[gamma[p]] = set()
            # TODO: whey gamma[p] itself is in adj_s??
            if a == gamma[p]:
                continue
            M[gamma[p]].add(a)
            M[a].add(gamma[p])

    t2 = datetime.now()
    print("Finish in {} seconds!".format((t2-t1).total_seconds()))

    # merge M 
    merge_info = {}
    for (a, b) in merge_history:
        if a not in merge_info:
            merge_info[a] = []
        
        merge_info[a].append(b)
        if b in merge_info:
            merge_info[a].extend(merge_info[b])
            merge_info.pop(b)

    with open("merge_info.json", 'w') as outfile:
        json.dump(merge_info, outfile, indent=4)
    
    with open("merge_history.json", 'w') as outfile:
        json.dump(merge_history, outfile, indent=4)

    for key, val in merge_info.items():    
        for v in val:
            for s in M[v]:
                M[s].remove(v)
                M[s].add(key)
            M[key].update(M[v])
            M.pop(v)

    # merge cc_table information
    new_S = {}
    for key, p_list in S.items():
        result = []
        cc1 = None
        for i, p in enumerate(p_list):
            if i == 0:
                cc1 = sorted(cc_table[p])
                result.append((p, tuple(cc1)))
            else:    
                cc2 = sorted(cc_table[p])
                for cc2_p in permutations(cc2, len(cc2)):
                    res = all(c1 & c2 for c1, c2 in zip(cc1, cc2_p))
                    if res:
                        result.append((p, tuple(cc2_p)))

        """
        order = None
        for i, p in enumerate(p_list):
            cc = sorted(cc_table[p])
            if i == 0:
                result.append((p, tuple(cc)))
                order = cc
            else:
                res = []
                for o in order:
                    for c in cc:
                        if c & o:
                            res.append(c)
                            break
                result.append((p, tuple(res)))
        """
        new_S[key] = result
    S = new_S
    
    with open("S{}.pkl".format(version), 'wb') as outfile:
        pickle.dump(S, outfile)

    with open("S{}.json".format(version), 'w') as outfile:
        S = {key:str(val) for key, val in S.items()}
        json.dump(S, outfile, indent=4)
    with open("M{}.json".format(version), 'w') as outfile:
        M = {key:str(val) for key, val in M.items()}
        json.dump(M, outfile, indent=4)
    with open("transition_table{}.json".format(version), 'w') as outfile:
        transition_table = {str(key):str(val) for key, val in transition_table.items()}
        #print(transition_table)
        json.dump(transition_table, outfile, indent=4)
    with open("gamma{}.json".format(version), 'w') as outfile:
        gamma = {str(key):val for key, val in gamma.items()}
        json.dump(gamma, outfile, indent=4)

def cc(graph, p, cc_table):
    if p in cc_table:
        return cc_table[p]

    set_p = set(p)
    nodes = [i for i in graph.keys() if i not in set_p]

    # build cc
    cc_list = []
    while nodes:
        original_node = nodes.pop()
        container = [original_node]
        visited = set()
        alive = set(container)
        while container:
            node = container.pop()
            visited.add(node)
            alive.remove(node)
            for neighbor in graph[node]:
                if neighbor not in alive and neighbor not in visited and neighbor not in set_p:
                    container.append(neighbor)
                    alive.add(neighbor)
        cc_list.append(visited)
        nodes = [n for n in nodes if n not in visited]

    cc_list = sorted(cc_list)
    cc_table[p] = cc_list

    return cc_list

def transition(cc1, cc2):
    #if len(cc1) != len(cc2):
    #    return False

    trans = []
    for c1 in cc1:
        for c2 in cc2:
            if c1 & c2:
                trans.append((c1, c2))

    return trans

    """
    # check bijection
    final_res = False
    for cc2_p in permutations(cc2, len(cc2)):
        res = all(c1 & c2 for c1, c2 in zip(cc1, cc2_p))
        if res:
            if not final_res:
                final_res = True
            elif final_res:
                print("two")
                return False
    return final_res 

    for c1, c2 in zip(cc1, cc2):
        if not(c1 & c2):
            return False
    return True

    index = [i for i, _ in enumerate(cc2)]
    for c1 in cc1:
        res = False
        for i in index:
            if c1 & cc2[i]:
                res = True
                index.remove(i)
                break
        if not res:
            return False
    return True
    
    # check bijection
    for cc2_p in permutations(cc2, len(cc2)):
        res = all(c1 & c2 for c1, c2 in zip(cc1, cc2_p))
        if res:
            return True

    return False
    """

# trainsition
def cc_similiar(graph, p1, p2, transition_table, cc_table):
    cc1 = cc(graph, p1, cc_table)
    cc2 = cc(graph, p2, cc_table)

    trans = transition(cc1, cc2)
    key = tuple(sorted([p1, p2]))
    transition_table[key] = trans
    
    # every component has a mapping
    if len(cc1) != len(cc2):
        return False

    # many-to-many mapping
    if len(trans) != len(cc1):
        return False
    
    # exist only once
    c1_set = set()
    c2_set = set()
    for c1, c2 in trans:
        if c1 in c1_set:
            return False
        else:
            c1_set.add(tuple(c1))

        if c2 in c2_set:
            return False
        else:
            c2_set.add(tuple(c2))

    if len(c1_set) == len(cc1) and len(c2_set) == len(cc2):
        return True
    else:
        return False

    """
    # length does not match
    if len(cc1) != len(cc2):
        return False

    # update transition
    trans = transition(cc1, cc2)
    if trans:
        if p1 not in transition_table:
            transition_table[p1] = set()
        transition_table[p1].add(p2)
        if p2 not in transition_table:
            transition_table[p2] = set()
        transition_table[p2].add(p1)

    return trans

    # equivalence relation
    if trans:
        return True
    else:
        return False
        # search
        #print("cc_similiar search")
        container = [p1]
        visited = set()
        alive = set(container)
        while container:
            p = container.pop()
            if p == p2:
                return True
            visited.add(p)
            alive.remove(p)
            for next_p in transition_table.get(p, []):
                if next_p not in alive and next_p not in visited:
                    container.append(next_p)
                    alive.add(next_p)
        #print("cc_similiar false")
        return False
    """

def search(num_p, name=""):
    from pprint import pprint

    # load data
    #graph = load_graph("state/{}.json".format(name))
    graph = create_simple_graph()
    
    # "4": "[((2, 4), ({5, 6}, {3}, {0, 1}))]",
    # "0": "[((0, 1), ({2, 3, 4, 5, 6},)), ((1, 0), ({2, 3, 4, 5, 6},)), ((0, 0), ({1, 2, 3, 4, 5, 6},))]",
    """
    with open("S{}.json".format(name), 'r') as infile:
        S = json.load(infile)
        S = {
            int(key) : [
                tuple(
                    int(vv) for vv in v.split(", "))
                )
                for v in val[3:-3].split(",)), ((")
            ]
            for key, val in S.items()
        }
    """
    with open("S{}.pkl".format(name), 'rb') as infile:
        S = pickle.load(infile)

    # "0": "{32, 1}",
    with open("M{}.json".format(name), "r") as infile:
        M = json.load(infile)
        M = {
            int(key) : [int(vv) for vv in val[1:-1].split(", ")]
            for key, val in M.items()         
        }

    # "(5, 6)": "{(6, 6)}",
    # "((2, 5), (2, 5))": "[({6}, {6}), ({3, 4}, {4}), ({0, 1}, {0, 1, 2})]",
    with open("transition_table{}.json".format(name), 'r') as infile:
        transition_table = json.load(infile)
        transition_table = {
            tuple(
                tuple(int(kk) for kk in k.split(", "))
                for k in key[2:-2].split("), (")
            ) : [
                tuple(
                    set(int(vvv) for vvv in vv.split(", "))
                    for vv in v.split("}, {")
                )
                for v in val[3:-3].split("}), ({")
            ]
            for key, val in transition_table.items()        
        }

    # "(4, 5)": 16,
    with open("gamma{}.json".format(name), 'r') as infile:
        gamma = json.load(infile)
        gamma = {
            tuple(int(k) for k in key[1:-1].split(", ")) : val
            for key, val in gamma.items()
        }

    # BFS
    visited = {}
    alive = set()
    cc_table = {}
    initial_p = tuple(0 for i in range(0, num_p))
    initial_abs = gamma[initial_p]
    initial_region = []
    real_state, cc = S[initial_abs][0]
    initial_state = ((initial_abs, (0,)), None)
    print(initial_state)

    container = deque([initial_state])
    alive.add(initial_state)

    while container:
        node, parent = container.popleft()
        visited[node] = parent
        state, region = node
        #print(node)

        # check goal
        if not region:
            print(node)
            print("Goal")
            
            # find path
            path = [node]
            current = node
            while True:
                parent = visited[current]
                if parent is None:
                    break
                path.append(parent)
                current = parent
            
            return path[::-1]

        # children
        children = get_next_children(M, S, graph, state, region)

        for child in children:
            if child not in alive and child not in visited:
                container.append((child, node))
    
    print("No Answer")

def get_next_children(M, S, graph, state, region):
    children = []
    for next_state in M[state]:
        p1, cc1 = S[next_state][0]
        next_region = []
        for i, c1 in enumerate(cc1):
            for r in region:
                for p2, cc2 in S[state]:
                    if not c1 & cc2[r]:
                        if next_state == 79:
                            print(c1, cc2, r)
                        break
                else:
                    next_region.append(i)
                    break
        children.append((next_state, tuple(next_region)))
        if next_state == 79:
            print(next_region)
            print(state, region, next_state)
    return children

def debug():
    import cv2
    import numpy as np

    position = [
        (10, 10),
        (20, 10),
        (30, 10),
        (40, 10),
        (50, 10),
        (10, 30),
        (20, 30),
        (30, 30),
        (40, 30),
        (50, 30),
        (10, 20),
        (20, 20),
        (40, 20),
        (50, 20),
    ]
    name = ""
    target_state = 22
    scale = 5
    height = 40*scale
    width = 60*scale
    unit = 5*scale
    position = [(a*scale, b*scale) for a, b in position]
    print(position)
    font = cv2.FONT_HERSHEY_SIMPLEX

    with open("S{}.pkl".format(name), 'rb') as infile:
        S = pickle.load(infile)
    
    print(len(S[22]))
    for count, (p, cc) in enumerate(S[22]):
        img = np.zeros((height, width), np.uint8)
        img += 255

        for index, pos in enumerate(position):
            if index in p:
                cv2.circle(img, pos, unit, 100, -1)
                cv2.putText(img, str(index), pos, font, 1, 220, 2)
            cv2.circle(img, pos, unit, 0, 1)

        cv2.imwrite("temp/{}_{}_{}.png".format(target_state, count, str(p)), img)

def main():
    #simple_graph = create_simple_graph()
    #abstract(simple_graph, 2)
    
    ladder_graph = create_Ladder_graph()
    abstract(ladder_graph, 4)
    
    #window_graph = create_window_graph()
    #abstract(window_graph, 4)
    
    #filename = "state/_ladder_k2_w2_state.json"
    #graph = load_graph(filename)
    #abstract(graph, 5)

    #result = cc(simple_graph, (2, 2))
    #print(result)

    path = search(4, "")
    pprint(path)
    #debug()

if __name__ == "__main__":
    main()
