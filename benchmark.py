import copy
import collections
import json
import pprint
import time

def generate_next_move(g, key, dirty, prs):
    # avoid duplicate
    d = {}

    # try to move each pursuer
    for i, pr in enumerate(prs):

        # find next position of pursuer
        for nxt_pr in g[pr]:

            # update next pursers
            nxt_prs = copy.deepcopy(prs)
            nxt_prs[i] = nxt_pr
            nxt_prs.sort()

            # update contaminated area of next dirty
            nxt_dirty = copy.deepcopy(dirty)
            nxt_dirty[nxt_pr] = 0
            update_contaminated_area(g, nxt_dirty, nxt_prs)
            nxt_key = ','.join([str(num) for num in nxt_dirty + nxt_prs])
            d[nxt_key] = (nxt_dirty, nxt_prs, tuple(prs))

    # return res
    res = []
    for s, tu in d.items():
        res.append((s, tu[0], tu[1], tu[2]))
    return res


def update_contaminated_area(g, dirty, prs):
    # init
    n = len(g)

    # add the pursers into visited et
    vst = set()
    for pr in prs:
        vst.add(pr)

    # iterate each node
    for node in range(n):
        if node not in vst and dirty[node]:
            update_contaminated_area_dfs(g, node, dirty, vst)


def update_contaminated_area_dfs(g, node, dirty, vst):
    # add to visited set
    vst.add(node)
    dirty[node] = 1

    # go through adjacent node
    for nxt_node in g[node]:
        if nxt_node not in vst:
            update_contaminated_area_dfs(g, nxt_node, dirty, vst)


def get_cc(g, dirty):
    # init
    n = len(g)

    ok = []
    bad = []

    vst = set()

    # iterate through the node
    for node in range(n):
        if node not in vst:
            cur_nodes = []
            if dirty[node]:
                cc_dfs(g, dirty, dirty[node], cur_nodes, node, vst)
                cur_nodes.sort()
                bad.append(cur_nodes)
            else:
                cc_dfs(g, dirty, dirty[node], cur_nodes, node, vst)
                cur_nodes.sort()
                ok.append(cur_nodes)
    ok.sort()
    bad.sort()
    return [ok, bad]


def cc_dfs(g, dirty, target, cur_nodes, node, vst):
    vst.add(node)
    cur_nodes.append(node)

    for nxt_node in g[node]:
        if nxt_node not in vst and dirty[nxt_node] == target:
            cc_dfs(g, dirty, target, cur_nodes, nxt_node, vst)


def is_cc_similar(cc1, cc2):
    ok1, bad1 = cc1
    ok2, bad2 = cc2

    # total size not equal
    if len(ok1) + len(bad1) != len(ok2) + len(bad2):
        return False
    if len(ok1) != len(ok2):
        return False
    if len(bad1) != len(bad2):
        return False

    # check ok list
    for l1 in ok1:
        st1 = set(l1)
        for l2 in ok2:
            st2 = set(l2)
            if st1 & st2:
                break
        else:
            return False
    for l1 in ok2:
        st1 = set(l1)
        for l2 in ok1:
            st2 = set(l2)
            if st1 & st2:
                break
        else:
            return False

    # check bad list
    for l1 in bad1:
        st1 = set(l1)
        for l2 in bad2:
            st2 = set(l2)
            if st1 & st2:
                break
        else:
            return False
    for l1 in bad2:
        st1 = set(l1)
        for l2 in bad1:
            st2 = set(l2)
            if st1 & st2:
                break
        else:
            return False

    return True


def main():

    t1 = time.time()

    # T graph
    n = 4
    g = collections.defaultdict(list)
    g[1].append(0)
    g[0].append(1)
    g[1].append(2)
    g[2].append(1)
    g[1].append(3)
    g[3].append(1)
    dirty = [0] * n
    dirty[1] = dirty[2] = dirty[3] = 1
    prs = [0, 0]
    key = ','.join([str(num) for num in dirty + prs])

    # window
    g = {
        0: [
            1,
            5
        ],
        1: [
            0,
            2
        ],
        2: [
            1,
            3,
            6
        ],
        3: [
            2,
            4
        ],
        4: [
            3,
            7
        ],
        5: [
            0,
            8
        ],
        6: [
            2,
            10
        ],
        7: [
            4,
            12
        ],
        8: [
            5,
            9
        ],
        9: [
            8,
            10
        ],
        10: [
            6,
            9,
            11
        ],
        11: [
            10,
            12
        ],
        12: [
            7,
            11
        ]
    }
    dirty = [1] * len(g)
    dirty[0] = 0
    prs = [0, 0, 0]
    key = ','.join([str(num) for num in dirty + prs])

    # parent
    parent = (-1, -1)

    # init abstract state connection
    abs_state_g = collections.defaultdict(set)
    abs_state_id_with_keys = collections.defaultdict(set)
    key_absstate = {}

    state_n = 0

    # init queue
    q = [(key, dirty, prs, parent)]

    # visited dirty state
    vst = set()

    # while queue is not empty
    while q:
        print('q ==============')
        print('q', q)
        print('q ==============')

        next_q = []
        for key, dirty, prs, prs_parent in q:
            # if ket already visited
            if key in vst:
                continue

            # put it into vst
            vst.add(key)

            # set abs state to null
            key_absstate[key] = None

            # adjs
            adjs = []

            # find next board by moving pursers
            for nxt_key, nxt_dirty, nxt_prs, nxt_prs_parent in generate_next_move(g, key, dirty, prs):
                if nxt_key in vst:
                    cc = get_cc(g, dirty)
                    nxt_cc = get_cc(g, nxt_dirty)
                    if is_cc_similar(cc, nxt_cc):
                        if key_absstate[key] is None:
                            nxt_state_id = key_absstate[nxt_key]
                            key_absstate[key] = nxt_state_id
                            abs_state_id_with_keys[nxt_state_id].add(key)

                        else:
                            # find the minimum
                            state_id, nxt_state_id = key_absstate[key], key_absstate[nxt_key]
                            if state_id == nxt_state_id:
                                continue

                            # print('Update Abstract Graph Conflict >>>>>>>>>>>>>')
                            # pprint.pprint(abs_state_g)
                            # print('Update Abstract Graph Conflict +++++++++++++')

                            # remove nxt_state_id to update abs_state_g
                            abs_state_g[state_id] = abs_state_g[state_id] | abs_state_g[nxt_state_id]
                            abs_state_g[nxt_state_id] = set()
                            for st_id, adj_st_ids in abs_state_g.items():
                                abs_state_g[st_id] = set(filter(lambda num: num != nxt_state_id, adj_st_ids))

                            # pprint.pprint(abs_state_g)
                            # print('Update Abstract Graph Conflict <<<<<<<<<<<<<')
                            # print()
                            # print()

                            # update key_absstate & abs_state_id_with_keys
                            for tmp_key, st_id in key_absstate.items():
                                if st_id == nxt_state_id:
                                    key_absstate[tmp_key] = state_id
                            abs_state_id_with_keys[state_id] = abs_state_id_with_keys[state_id] \
                                                               | abs_state_id_with_keys[nxt_state_id]
                            abs_state_id_with_keys[nxt_state_id] = set()

                    else:
                        adjs.append(key_absstate[nxt_key])

                else:
                    next_q.append((nxt_key, nxt_dirty, nxt_prs, nxt_prs_parent))

            # update p abstract state
            if key_absstate[key] is None:
                state_id = state_n
                state_n += 1
                key_absstate[key] = state_id
                abs_state_id_with_keys[state_id].add(key)

            # update connections in state graph
            # print('Update Abstract Graph >>>>>>>>>>>>>')
            # pprint.pprint(abs_state_g)
            # print('Update Abstract Graph +++++++++++++')
            if key_absstate[key] not in abs_state_g:
                abs_state_g[key_absstate[key]] = set()
            for adj_state in adjs:
                abs_state_g[key_absstate[key]].add(adj_state)
                abs_state_g[adj_state].add(key_absstate[key])
            # pprint.pprint(abs_state_g)
            # print('Update Abstract Graph <<<<<<<<<<<<<')
            # print()
            # print()

        q = next_q

    t2 = time.time()

    pprint.pprint(abs_state_g)
    print(t2 - t1)
    # pprint.pprint(abs_state_id_with_keys)
    # print(json.dumps(key_absstate, indent=4))
    # print(len(key_absstate))


def test():
    # test generate_next_board()
    print('test generate_next_board()')
    n = 4
    g = collections.defaultdict(list)
    g[1].append(0)
    g[0].append(1)
    g[1].append(2)
    g[2].append(1)
    g[1].append(3)
    g[3].append(1)
    dirty = [0] * n
    dirty[1] = dirty[2] = dirty[3] = 1
    key = ''.join([str(num) for num in dirty])
    prs = [0, 0, 0]
    res = generate_next_move(g, key, dirty, prs)
    print(res)
    print()

    # test cc()
    print('test cc()')
    n = 4
    g = collections.defaultdict(list)
    g[1].append(0)
    g[0].append(1)
    g[1].append(2)
    g[2].append(1)
    g[1].append(3)
    g[3].append(1)
    dirty = [0] * n
    dirty[0] = dirty[2] = dirty[3] = 1
    key = ''.join([str(num) for num in dirty])
    prs = [0, 0]
    print(get_cc(g, dirty))
    print()

    # test is_cc_similar()
    print('test is_cc_similar()')
    n = 4
    g = collections.defaultdict(list)
    g[1].append(0)
    g[0].append(1)
    g[1].append(2)
    g[2].append(1)
    g[1].append(3)
    g[3].append(1)

    #  X
    #  O X
    #  X
    dirty1 = [0] * n
    dirty1[0] = dirty1[2] = dirty1[3] = 1
    cc1 = get_cc(g, dirty1)

    #  O
    #  X X
    #  X
    dirty2 = [0] * n
    dirty2[1] = dirty2[2] = dirty2[3] = 1
    cc2 = get_cc(g, dirty2)

    #  X
    #  X O
    #  X
    dirty3 = [0] * n
    dirty3[0] = dirty3[1] = dirty3[2] = 1
    cc3 = get_cc(g, dirty3)

    #  X
    #  X X
    #  O
    dirty4 = [0] * n
    dirty4[0] = dirty4[1] = dirty4[3] = 1
    cc4 = get_cc(g, dirty4)

    # print('cc1', 'cc2', is_cc_similar(cc1, cc2))
    # print('cc1', 'cc3', is_cc_similar(cc1, cc3))
    # print('cc1', 'cc4', is_cc_similar(cc1, cc4))
    print('cc2', 'cc3', is_cc_similar(cc2, cc3))
    print('cc2', 'cc4', is_cc_similar(cc2, cc4))
    print('cc3', 'cc4', is_cc_similar(cc3, cc4))


if __name__ == '__main__':
    main()
