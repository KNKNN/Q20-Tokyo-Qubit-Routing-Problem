# Q20-Tokyo-Qubit-Routing-Problem
NTHU CS undergraduate research

## Introduction
Solving the coupling constraint of CNOT gate in quantum circuits. 
The routing problem is composed of two main parts which are generating initial mapping and 
generating SWAP gates to insert into each time step while the circuit is executing.

## Features
* Using **Simulated Annealing (SA)** method to generate initial mapping.
* Using a **Heuristic method** to evaluate the cost and choose SWAP gates.

## How to start 
1. Copy the quantum circuit data into `circuit.txt`
2. Run :  `python3 quantum_1027.py`
3. The initial mapping result and the chosen swap gates will directly print on the screen. 

## Result
* Initial Mapping
* Time(s)
* Gate added (gadd)
* Depth
![image](https://user-images.githubusercontent.com/18658857/144086272-e55c3ff0-ec7e-462c-90db-522f381a3963.png)

## Ideas to improve
1. using reinforcement learning method (Deep Q-Learning) to find SWAP gates instead
of using Heuristic method.
