ASU Planning Final Project
--------------------------

Contents:

    1) How to run "benchmark_brute_force.py"
    2) How to run paper "benchmark.py"
    3) How to generate the graph


1) How to run "benchmark_brute_force.py"
-----------------------------------------
Go to the submission folder:
>> cd submission
>> python3 benchmark_brute_force.py state/_tree_k1_w1_state.json

You will see a output file ans__tree_k1_w1_state.txt in the same directory


state/_tree_k1_w1_state.json
V = 10
No ans for N = 1
Ans for N = 2
((0, 0), (0, 1, 1, 1, 1, 1, 1, 1, 1, 1))
((1, 0), (0, 0, 1, 1, 1, 1, 1, 1, 1, 1))
((2, 0), (0, 0, 0, 1, 1, 1, 1, 1, 1, 1))
((3, 0), (0, 0, 0, 0, 1, 1, 1, 1, 1, 1))
((4, 0), (0, 1, 1, 1, 0, 1, 1, 1, 1, 1))
((4, 1), (0, 0, 1, 1, 0, 1, 1, 1, 1, 1))
((4, 2), (0, 0, 0, 1, 0, 1, 1, 1, 1, 1))
((4, 3), (0, 0, 0, 0, 0, 1, 1, 1, 1, 1))
((5, 3), (0, 0, 0, 0, 0, 0, 1, 1, 1, 1))
((5, 7), (0, 0, 0, 0, 0, 0, 1, 0, 1, 1))
((6, 7), (0, 0, 0, 0, 0, 0, 0, 0, 1, 1))
((6, 8), (0, 0, 0, 0, 0, 0, 0, 0, 0, 1))
((6, 9), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
path len = 13
Counter = 249
0.005872 s
