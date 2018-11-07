ASU Planning Final Project
--------------------------


1) How to run "brute_force.py"
------------------------------
Go to the submission folder:

```
>> cd submission
>> python3 brute_force.py state/_tree_k1_w1_state.json
```
If no json filename given in last command, the default graph would be looks like this

```
    0
    |
    0
    |
    0-0-0
    |
    0
    |
    0
``` 

After execution, you will see a output file ans_brute_force_tree_k1_w1_state.txt in the same directory contains the result.

```
state/_tree_k1_w1_state.json
Number of vertices in graph: 10
Minimum pursuers is N = 2
Shortest path start: 
	(0, 0)
	(1, 0)
	(2, 0)
	(3, 0)
	(4, 0)
	(4, 1)
	(4, 2)
	(4, 3)
	(5, 3)
	(5, 7)
	(6, 7)
	(6, 8)
	(6, 9)
Path length is 13
Time elapsed: 0.00677 s

```

2) How to run our "benchmark.py"
--------------------------------
Go to the submission folder:

```
>> cd submission
>> python3 benchmark.py state/_tree_k1_w1_state.json
```

After execution, you will see a output file ans_benchmark_tree_k1_w1_state.txt in the same directory contains the result.


3) How to draw maps and figures:
-----------------------------
We use paraneters in fig_config.txt to generate all figures.

We already generated all maps needed. Please DO NOT run this file.

