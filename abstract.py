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
        cc1 = cc(graph, p, cc_table)

        for next_p in children:
            if next_p in visited:
                if gamma[p] == gamma[next_p]:
                    continue
                
                cc2 = cc(graph, next_p, cc_table)
                sim, trans, reverse_trans = cc_similiar(graph, p, next_p, cc1, cc2, transition_table, cc_table)
                if sim:
                    if gamma[p] is None:
                        gamma[p] = gamma[next_p]
                        cc_order = S[gamma[next_p]][next_p]
                        try:
                            new_cc = tuple(reverse_trans[cc][0] for cc in cc_order)
                        except KeyError as error:
                            print("key error", error)
                            print(trans)
                            print(reverse_trans)
                            print(cc_order)
                            quit()
                        S[gamma[p]][p] = new_cc
                        
                    else:
                        # resolve conflict
                        abs1 = gamma[p]
                        abs2 = gamma[next_p]
                        
                        # merge S
                        cc_order = S[abs1][p]
                        new_order = S[abs2][next_p]
                        mapping = []
                        for c in cc_order:
                            mapping.append(new_order.index(trans[c][0]))
                        
                        if not all(i == m for i, m in enumerate(mapping)):
                            S[abs2] = {
                                p : tuple(c[m] for m in mapping)
                                for p, c in S[abs2].items()
                            }

                        S[abs1].update(S[abs2])
                        for temp_p in S[abs2].keys():
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
            
            S[abs_p] = {p : tuple(tuple(sorted(c)) for c in cc1)}
            #print(S[abs_p])
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
    S = {
        key : tuple((p, tuple(set(c) for c in c_list)) for p, c_list in val.items())
        for key, val in S.items()        
    }
    """
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
        new_S[key] = result
    S = new_S
    """
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

    cc1 = [(c, tuple(sorted(c))) for c in cc1]
    cc2 = [(c, tuple(sorted(c))) for c in cc2]

    trans = {}
    reverse_trans = {}
    count = 0
    for c1, c1_t in cc1:
        for c2, c2_t in cc2:
            if c1 & c2:
                count += 1
                
                if c1_t not in trans:
                    trans[c1_t] = []
                trans[c1_t].append(c2_t)
                
                if c2_t not in reverse_trans:
                    reverse_trans[c2_t] = []
                reverse_trans[c2_t].append(c1_t)

    return trans, reverse_trans, count

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
def cc_similiar(graph, p1, p2, cc1, cc2, transition_table, cc_table):
    #cc1 = cc(graph, p1, cc_table)
    #cc2 = cc(graph, p2, cc_table)

    trans, reverse_trans, count = transition(cc1, cc2)
    key = tuple(sorted([p1, p2]))
    transition_table[key] = trans
    
    """
    print(trans)
    print(reverse_trans)
    print(len(trans))
    print(len(reverse_trans))
    print(len(cc1))
    print(count)
    print(len(cc1))
    """

    # every component has a mapping
    if len(cc1) != len(cc2):
        return False, None, None

    # remove many-to-many mapping
    if count != len(cc1):
        return False, None, None
    
    # exist only once
    if len(trans) == len(reverse_trans) == len(cc1):
        return True, trans, reverse_trans
    
    return False, None, None

    """
    c1_set = set()
    c2_set = set()
    for c1, c2 in trans:
        if c1 in c1_set:
            return False, None
        else:
            c1_set.add(c1)

        if c2 in c2_set:
            return False, None
        else:
            c2_set.add(c2)

    if len(c1_set) == len(cc1) and len(c2_set) == len(cc2):
        return True, trans
    else:
        return False, None

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

def search(num_p, filename, name=""):
    from pprint import pprint

    # load data
    graph = load_graph(filename)
    #graph = create_simple_graph()
    #graph = create_simple_h_graph()
    #graph = create_Ladder_graph()
    
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
    """
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
    """
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
            print("node = ", node, "parent = ", parent)
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
            print("ABS path = ", path[::-1])
            refinement(S, M, graph, gamma, path[::-1], num_p)

            return path[::-1]

        # children
        children = get_next_children(M, S, graph, state, region)

        for child in children:
            if child not in alive and child not in visited:
                container.append((child, node))
                alive.add(child)
    
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
                        if next_state == 3 and state == 22:
                            print("=========================")
                            print("c1, next {}, ".format(next_state), c1)
                            print(p1)
                            print()
                            print("cc2, current {}, ".format(state), cc2[r])
                            print(p2)
                            print("r", r)
                            print("=========================")
                        break
                else:
                    next_region.append(i)
                    break
        children.append((next_state, tuple(next_region)))
    
    return children

def refinement(S, M, graph, gamma, path, num_p):

    total_path = []
    initial_p = tuple(0 for i in range(0, num_p))

    for S1, S2 in zip(path[:-1], path[1:]):
        abs1, L1 = S1
        abs2, L2 = S2
        path = refine_in_abs(S, M, graph, gamma, abs1, L1, abs2, L2, initial_p, num_p)
        total_path.append(path)
        initial_p = path[-1]
        print(path)
    print("========== refinement result ===========")
    pprint(total_path)

def build_real_state(p, cc_list, L2, graph):
    dirty_region = {c for l in L2 for c in cc_list[l]}
    c = tuple(1 if g in dirty_region else 0 for g in graph.keys())
    return (p, c)

def refine_in_abs(S, M, graph, gamma, abs1, L1, abs2, L2, initial_p, num_p):
    visited = {}
    alive = set()
    #initial_p = tuple(0 for i in range(0, num_p))
    initial_abs = gamma[initial_p]
    initial_region = []
    initial_state = (initial_p, None)
    #print(initial_state)

    current_real_set = {p:cc_list for p, cc_list in S[abs1]}
    target_real_set = [
        build_real_state(p, cc_list, L2, graph)
        for p, cc_list in S[abs2]
    ]
    target_real_set = {
        p : s
        for p, s in target_real_set
    }
    #pprint(current_real_set)
    #pprint(target_real_set)
    
    container = deque([initial_state])
    alive.add(initial_state)
    while container:
        node, parent = container.popleft()
        #print("current node = ", node)

        # check goal if node is in target_real_set
        if node in target_real_set:
            #print(node, "in target. parent =", parent)
            #print(target_real_set)
            # generate contaminated for parent
            dirty_region = {c for l in L1 for c in current_real_set[parent][l]}
            dirty_region = tuple(1 if g in dirty_region else 0 for g in graph.keys())
            new_dirty = contaminate(parent, node, dirty_region, graph)
            
            #print(dirty_region)
            #print(new_dirty)
            #print(target_real_set[node])

            if new_dirty == target_real_set[node]:
                # find path
                path = [node, parent]
                current = parent
                while True:
                    parent = visited[current]
                    if parent is None:
                        break
                    path.append(parent)
                    current = parent
                #print(path[::-1])
                return path[::-1]
        
        # check available
        if node not in current_real_set:
            continue
        
        visited[node] = parent
        children = {
            tuple(ppp if i != ii else next_p for ii, ppp in enumerate(node))
            for i, pp in enumerate(node)
                for next_p in graph[pp]
        }
        #print(children)

        for child in children:
            if child not in alive and child not in visited:
                container.append((child, node))
                alive.add(child)
    print("GG refinement path not found between {} & {}".format(str(abs1), str(abs2)))

def contaminate(p, ps, dirty, graph):
    #print("contaminate")

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
    target_state = 45
    scale = 5
    height = 40*scale
    width = 60*scale
    unit = 5*scale
    position = [(a*scale, b*scale) for a, b in position]
    print(position)
    font = cv2.FONT_HERSHEY_SIMPLEX

    with open("S{}.pkl".format(name), 'rb') as infile:
        S = pickle.load(infile)
    
    print("s size = ", len(S[target_state]))
    for count, (p, cc) in enumerate(S[target_state]):
        img = np.zeros((height, width), np.uint8)
        img += 255

        for index, pos in enumerate(position):
            if index in p:
                cv2.circle(img, pos, unit, 100, -1)
                cv2.putText(img, str(index), pos, font, 1, 220, 2)
            cv2.circle(img, pos, unit, 0, 1)

        cv2.imwrite("temp/{}_{}_{}.png".format(target_state, count, str(p)), img)

def main():
    #debug()
    #quit()

    num_p = 2
    #simple_graph = create_simple_graph()
    #abstract(simple_graph, num_p)

    #simple_graph = create_simple_h_graph()
    #abstract(simple_graph, 2)

    #ladder_graph = create_Ladder_graph()
    #abstract(ladder_graph, num_p)
    
    #window_graph = create_window_graph()
    #abstract(window_graph, 4)
    
    #filename = "state/_ladder_k3_w1_state.json"
    filename = "state/_tree_k2_w1_state.json"
    graph = load_graph(filename)
    abstract(graph, num_p)

    #result = cc(simple_graph, (2, 2))
    #print(result)

    path = search(num_p, filename, "")
    pprint(path)

if __name__ == "__main__":
    main()
