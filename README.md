ASU Planning Final Project
--------------------------

# Requirement

* Python 3.5+
* Library: numpy, pickle, opencv-python, imageio

#Extension
0) How to run reinforcement_learning.py
---------------------------------------

The program has 2 phases. 
You can either train or test a model.

### For training:

Run the following command

```
>> python3 reinforcement_learning.py [-p training] [-m maptype] [-k branch_number] [-w width] [-n pursuer]
```

With following options.

```
-m: choosing map from [ladder, tree] for training. (Default:ladder)
-k: choosing branch number(Default:3)
	Tree:   1-4
	ladder: 2-4	
-w: choosing width with option of 1 or 2
-n: number of pursuer. 
	Please reference to pursuer.txt for minimum number of pursuers.
-p: training
	
```
Example:
>python3 reinforcement_learning.py -m ladder -k 3 -w 1 -n 3 -p training



### For testing:

After training, you will have a directary [ladder\_k3\_w1] in model.

you can test the model with the following command.


```
python3 reinforcement_learning.py [-p testing] [-m maptype] [-k branch_number] [-w width] [-n pursuer] [-d] [-e episode]
```

```
-p: testing
-m: choosing map from [ladder, tree] for training. (Default:ladder)
-k: choosing branch number(Default:3)
	Tree:   1-4
	ladder: 2-4	
-w: choosing width with option of 1 or 2
-n: number of pursuer. 
	Please reference to pursuer.txt for minimum num of pursuers.
-d: save testing result as a gif file
-e: multiple of 5000
```

Example:
>python3 reinforcement_learning.py -m ladder -k 3 -w 1 -n 3 -p testing -e 10000 -d


With -d enable, result of testing will be saved in ./demo


# Benchmark
1) How to run "brute_force.py"
------------------------------
Run the following command

```
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
Run the following command

```
>> python3 benchmark.py state/_tree_k1_w1_state.json
```

After execution, you will see a output file ans_benchmark_tree_k1_w1_state.txt in the same directory contains the result.


3) How to draw maps and figures:
-----------------------------
We use paraneters in fig_config.txt to generate all figures.

We already generated all maps needed. Please DO NOT run this file.

