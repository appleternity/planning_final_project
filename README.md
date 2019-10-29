# Reinforcement Learning for Solving the Pursuit-Evasion Problems
ASU CSE 574 Planning/Learning Methods AI Final Project 

[[Paper](https://raw.githubusercontent.com/appleternity/planning_final_project/3bf9a7745b3f97cbc36da8763ed7f38a7c3f352a/Planning_Final_Project.pdf)]
[[Slides](https://docs.google.com/presentation/d/1_0p_jFYE5NQ6rX_o51VKwKm59AnMBBm8661jEA7Ug3I/edit?usp=sharing)]
--------------------------
# Introduction

In this paper, we focused on the worst case of the pursuitevasion problem, where an unknown number of evaders is presented in the given environment. All of the evaders could move arbitrary fast and the movement is unpredictable. Our goal is to have a policy that can help the pursuers capture all of the evaders.

<p align="center">
	<img src="https://raw.githubusercontent.com/appleternity/planning_final_project/final_submission/demo/ladder_k2_w2_e30000/demo.gif">
	<br>
	<b>Figure 1. Example</b>
</p>

# Proposed Algorithm
## Q-Learning
Using Q-Learning for this problem we also set up a step limit n. If the agent can not clean the region within n steps, we treat it as a fail and give it a penalty. If the agent successfully clean up the region, then we give it a reward. 

## State
We used a -1 to represent a contaminated node, a 0 to represent a clear node, and a positive number to represent the number of the pursuers. Figure 2 shows two examples of our state representation. In such representation, the goal state is the state where -1 does not exist.

<p align="center">
	<img src="https://raw.githubusercontent.com/appleternity/planning_final_project/final_submission/demo/state_representation.png">
	<br>
	<b>Figure 2. Example of the state representation, where black stands for the pursuer, green stands for the clear node, and red stands for the dirty node. (a) {0; 0; 2; 0; 0; −1; −1} (b) {0; 0; 1; 1; 0; −1; −1} </b>
</p>

## Reward 
Reward can be split into two categories, the step reward and the terminal reward.

### Step Reward
To encourage the agent to clear region, we integrate both the number of the clear region and step cost into the step reward.

<img src="https://latex.codecogs.com/png.latex?%5Clarge%20Reward_%7Bstep%7D%20%3D%20Number%5C%20of%5C%20Clear%5C%20Region%5C%20-%20%5C%20Step%5C%20Cost">

### Terminal Reward
Three terminal rewards were tried. 

#### Fixed Reward
<img src="https://latex.codecogs.com/png.latex?%5Clarge%20R_%7Bf%7D%3D%20%5Cbegin%7Bcases%7D%20&plus;150%20%26%2C%20if%5C%20reaching%5C%20goal%20%5C%5C%20-150%20%26%2C%20otherwise%20%5Cend%7Bcases%7D">

#### Dynamic Reward
<img src="https://latex.codecogs.com/png.latex?%5Clarge%20R_%7Bd%7D%3D%20%5Cbegin%7Bcases%7D%2020%20%5Ccdot%20node_%7Bclear%7D%20%26%2C%20if%5C%20reaching%5C%20goal%20%5C%5C%20-1000%20&plus;%2010%20%5Ccdot%20node_%7Bclear%7D%20%26%2C%20otherwise%20%5Cend%7Bcases%7D">

#### Dynamic Reward + Boundary Reward
<img src="https://latex.codecogs.com/png.latex?%5Clarge%20R_%7Bb%7D%3D%20%5Cbegin%7Bcases%7D%2020%20%5Ccdot%20node_%7Bclear%7D%20%26%2C%20if%5C%20reaching%5C%20goal%20%5C%5C%20-1000%20&plus;%2010%20%5Ccdot%20node_%7Bclear%7D%20%5C%5C%20%5Cquad%5Cquad%5Cquad&plus;%20100%20%5Ccdot%20pursuer_%7Bboundary%7D%20%26%2C%20otherwise%20%5Cend%7Bcases%7D">

## Exploration Strategy
Three exploration strategies were used in this project.
(1) Purely Random
(2) Boltzmann-distributed Exploration
(3) Counter-based Exploration

# Requirement

* Python 3.5+
* Library: numpy, pickle, opencv-python, imageio

# Extension
0) How to run reinforcement_learning.py
---------------------------------------
1. Go to the submission folder:

```
>> cd submission
```

The program has 2 phases. 
You can either train or test a model.

### For training:

Use following command to execute training phase.

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
-d: save demo result		
-e: multiple of 5000
```

Example:
>python3 reinforcement_learning.py -m ladder -k 3 -w 1 -n 3 -p testing -e 5000 -d


With -d enable, result of testing will be saved in ./demo


# Benchmark
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

