from collections import defaultdict
from random import randint
import numpy as np
import random as rd
import copy
import time
import re
import math
import itertools

F = 99999 # INF = 99999
V = 20 # graph size
mapping = []
test_mapping = []

class QubitRouting:
    def __init__(self):        
        
        # connection of physical qubits (adjacency list) Q20 Tokyo 
        self.graph = {'0': ['1', '5'], 
                      '1': ['0', '2', '6'],
                      '2': ['1', '3', '6', '7'],
                      '3': ['2', '8', '4'],
                      '4': ['3', '8', '9'],
                      '5': ['0', '6', '10', '11'],
                      '6': ['1', '2', '5', '7', '10', '11'],
                      '7': ['1', '2', '6', '8', '12', '13'],
                      '8': ['3', '4', '7', '9', '12', '13'],
                      '9': ['3', '4', '8', '14'],
                      '10':['5', '6', '11', '15'],
                      '11':['5', '6', '10', '12', '16', '17'],
                      '12':['7', '8', '11', '13', '16', '17'],
                      '13':['7', '8', '12', '14', '18', '19'],
                      '14':['9', '13', '18', '19'],
                      '15':['10', '16'],
                      '16':['11', '12', '15', '17'],
                      '17':['11', '12', '16', '16', '18'],
                      '18':['13', '14', '17', '17', '19'],
                      '19':['13', '14', '18']}

        # unexecuted gates
        self.unexecuted_gates = []
        with open(r'C:\Users\Kn\Desktop\circuit.txt') as file:
            for line in file:
                for word in line.split():
                    if (word == 'cx'):
                        p = '[\d]+'
                        cnot_gate = []
                        if re.search(p, line) is not None:
                            for catch in re.finditer(p, line):
                                cnot_gate.append(int(catch[0]))
                        #print(cnot_gate)
                        self.unexecuted_gates.append(cnot_gate)

        # front layer of unexecuted gates
        self.unexecuted_gates_front_layer = []

        # number of swaps 
        self.swap_num = 0
        
        # total depth
        self.depth = 0

    def generate_gate(self, probability_of_gates, gate_appear_time_sorted, gate_combination):
        temp_probability_of_gates = copy.deepcopy(probability_of_gates)
        test_gate = []
        p = rd.random()
        cumulative_probability = 0.0

        for i in range(len(gate_combination)):
            cumulative_probability += temp_probability_of_gates[i]
            if p <= cumulative_probability:
                test_gate = gate_combination[gate_appear_time_sorted[i]][0]
                break
        
        return test_gate

    def BFS_SP(self, graph_in, start, goal):
        graph = copy.deepcopy(graph_in)

        for i in range(0, V):
            for j in range(len(graph[str(i)])):
                if str(i) not in graph[graph[str(i)][j]]:
                    graph[graph[str(i)][j]].append(str(i))

	    # Queue for traversing the graph in the BFS
        queue = [[start]]
        explored = []

	    # If the desired node is reached
        if start == goal:
            print("Same Node")
            return 99999
	
	    # Loop to traverse the graph with the help of the queue
        while queue:
            path = queue.pop(0)
            node = path[-1]
		
		    # Condition to check if the current node is not visited
            if node not in explored:
                neighbours = graph[node]
			
			    # Loop to iterate over the neighbours of the node
                for neighbour in neighbours:
                    new_path = list(path)
                    new_path.append(neighbour)
                    queue.append(new_path)
                    
				# Condition to check if the neighbour node is the goal
                    if neighbour == goal:
                        return len(new_path) - 2
                explored.append(node)

	    # Condition when the nodes are not connected
        print("So sorry, but a connecting path doesn't exist :(")

        return 99999

    # swap that doesn't change outside mapping
    def swap(self, mapping_in, q1, q2):
        mapping = copy.deepcopy(mapping_in)
        a = mapping[mapping.index(q1)]
        b = mapping[mapping.index(q2)]
        c = mapping.index(q1)
        d = mapping.index(q2)
        
        a, b = b, a
        mapping[c] = a
        mapping[d] = b

        return mapping
    
    # swap that change outside mapping
    def swap2(self, mapping, q1, q2):
        a = mapping[mapping.index(q1)]
        b = mapping[mapping.index(q2)]
        c = mapping.index(q1)
        d = mapping.index(q2)
        a, b = b, a
        mapping[c] = a
        mapping[d] = b 

    # generate swap candidates (adjacency list version)
    def generate_swap_candidate(self, graph_in):
        graph = copy.deepcopy(graph_in)
        swap_candidates = []

        for i in range(0, V):
            for j in range(len(graph[str(i)])):
                swap_candidates.append([i, int(graph[str(i)][j])])
                if str(i) in graph[graph[str(i)][j]]:
                    graph[graph[str(i)][j]].remove(str(i))
                
        return swap_candidates

    #calculate the remaining gates cost (adjacency list version)
    def calculate_per_swap_cost(self, graph, mapping, unexecuted_gates, swap):
        cost = 0
        mapping = self.swap(mapping, swap[0], swap[1])

        #calculate cost
        for i in range(len(unexecuted_gates)): 
            cost += self.BFS_SP(graph, str(mapping[unexecuted_gates[i][0]]), str(mapping[unexecuted_gates[i][1]]))
            
        return cost

    def generate_action(self, graph, mapping, unexecuted_gates_front_layer, swap_candidates_in):
        mapping_temp = copy.deepcopy(mapping)
        find_action = False
        prev_min = 99999
        temp_swap_candidates =  copy.deepcopy(swap_candidates_in)
        temp_swap_candidates_2 = copy.deepcopy(swap_candidates_in)
        swap_candidates_cost = []
        action = []
        chosen_swap = [0, 0]
         
        while find_action == False:
            swap_candidates_cost = []
            temp_swap_candidates = copy.deepcopy(temp_swap_candidates_2)
        
            while temp_swap_candidates != []:
                swap_candidates_cost.append(self.calculate_per_swap_cost(graph, mapping, unexecuted_gates_front_layer, temp_swap_candidates.pop(0)))
            #print("each swap cost:", swap_candidates_cost)
            
            # no better swap to choose
            if prev_min != 0 and prev_min <= min(swap_candidates_cost):
                find_action = True
                
            swap_candidates_cost_arr = np.array(swap_candidates_cost)
            sorted_swap = swap_candidates_cost_arr.argsort()
            
            x = np.random.randint(18)
            a = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
            #print("swap cost:", swap_candidates_cost)
            x = a[x]
            if min(swap_candidates_cost) == 0:
                gates_to_remove = []
                #print("FL:", unexecuted_gates_front_layer)

                for i in range (len(unexecuted_gates_front_layer)):
                    if str(mapping_temp[unexecuted_gates_front_layer[i][1]]) in graph[str(mapping_temp[unexecuted_gates_front_layer[i][0]])] or str(mapping_temp[unexecuted_gates_front_layer[i][0]]) in graph[str(mapping_temp[unexecuted_gates_front_layer[i][1]])]:
                        gates_to_remove.append(unexecuted_gates_front_layer[i])
                        #print("gates_to_remove", gates_to_remove)

                while gates_to_remove != []:
                    self.unexecuted_gates.remove(gates_to_remove[0])
                    self.unexecuted_gates_front_layer.remove(gates_to_remove.pop(0))
                        
                if unexecuted_gates_front_layer == []:
                    #print("fl emp")
                    find_action = True
                else:
                    self.swap2(mapping_temp, temp_swap_candidates_2[sorted_swap[x]][0], temp_swap_candidates_2[sorted_swap[x]][1])  
                    chosen_swap = temp_swap_candidates_2[sorted_swap[x]]
                    action.append(chosen_swap)
                    self.swap_num += 1
                    find_action = True
                    self.swap2(mapping_temp, temp_swap_candidates_2[sorted_swap[x]][0], temp_swap_candidates_2[sorted_swap[x]][1])
            else:
                self.swap2(mapping_temp, temp_swap_candidates_2[sorted_swap[x]][0], temp_swap_candidates_2[sorted_swap[x]][1]) 
                chosen_swap = temp_swap_candidates_2[sorted_swap[x]]
                action.append(chosen_swap)
                self.swap_num += 1
                self.swap2(mapping_temp, temp_swap_candidates_2[sorted_swap[x]][0], temp_swap_candidates_2[sorted_swap[x]][1])
            
            # remove the swaps with duplicate qubits to the chosen_swap
            candidates_to_remove = []
            
            for i in range(len(temp_swap_candidates_2)):
                #print(chosen_swap)
                if chosen_swap[0] == temp_swap_candidates_2[i][0] or chosen_swap[1] == temp_swap_candidates_2[i][0] or chosen_swap[0] == temp_swap_candidates_2[i][1] or chosen_swap[1] == temp_swap_candidates_2[i][1]:
                    candidates_to_remove.append(temp_swap_candidates_2[i])
                    
            for i in range(len(candidates_to_remove)):
                temp_swap_candidates_2.remove(candidates_to_remove.pop(0))
            
            # update the previous minimal swap cost
            prev_min = min(swap_candidates_cost)
            
        return action

    def operate_action(self, graph, mapping, action, unexecuted_gates_front_layer): 
        gates_to_remove = []

        if action == []:
            #print("hi", unexecuted_gates_front_layer)
            for i in range(len(unexecuted_gates_front_layer)):
                gates_to_remove.append(unexecuted_gates_front_layer[i])

            while gates_to_remove != []:
                #print("gates to remove:", gates_to_remove[0])
                self.unexecuted_gates.remove(gates_to_remove[0])
                self.unexecuted_gates_front_layer.remove(gates_to_remove.pop(0))

            return

        for i in range(len(action)):
            self.swap2(mapping, action[i][0], action[i][1])
        
        for i in range(len(unexecuted_gates_front_layer)):
            if self.BFS_SP(graph, str(mapping[unexecuted_gates_front_layer[i][0]]), str(mapping[unexecuted_gates_front_layer[i][1]])) == 0:# and (str(unexecuted_gates_front_layer[i][1]) in graph[str(unexecuted_gates_front_layer[i][0])])):
                gates_to_remove.append(unexecuted_gates_front_layer[i])
            
        while (gates_to_remove != []):
            self.unexecuted_gates.remove(gates_to_remove[0])
            self.unexecuted_gates_front_layer.remove(gates_to_remove.pop(0))
    
    def gencoordinates(self, m, n):
        seen = set()
        x, y = randint(m, n), randint(m, n)

        while True:
            seen.add((x, y))
            yield (x, y)
            x, y = randint(m, n), randint(m, n)
            while (x, y) in seen:
                x, y = randint(m, n), randint(m, n)

    def acceptance_probability(self, prev_score, new_score, temperature):
        if new_score < prev_score:
            return 1
        elif new_score == prev_score:
            #print(0.1)
            return 0.00001
        else :
            p = np.exp(-(new_score - prev_score) / temperature) # exp(-(E(Xnew) - E(Xold)) / temperature) -> Metropolis principle
            print("accept probability :", p)
            return p

    # SA algorithm
    def generate_mapping(self, mapping):
        # generate all the combination of gates on real circuit 
        gate_combination = []
        for pair in itertools.combinations([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], 2):
            gate_combination.append([[pair[0], pair[1]], 0])
            gate_combination.append([[pair[1], pair[0]], 0])

        
        for i in range(len(self.unexecuted_gates)):
            for j in range(len(gate_combination)):
                if (gate_combination[j][0] == self.unexecuted_gates[i]):
                    gate_combination[j][1] += 1
        
        gate_appear_time = []
        for i in range(len(gate_combination)):
            gate_appear_time.append(gate_combination[i][1])

        gate_appear_time_arr = np.array(gate_appear_time)
        gate_appear_time_sorted = np.argsort(-gate_appear_time_arr)


        mapping_count = []
        for _ in range (20):
            mapping_count.append(0)
            mapping.append(0)

        for i in range (len(self.unexecuted_gates)):
            mapping_count[self.unexecuted_gates[i][0]] += 1
            mapping_count[self.unexecuted_gates[i][1]] += 1

        mapping_count_arr = np.array(mapping_count)
        mapping_count_sorted = np.argsort(-mapping_count_arr)

        mapping_order = [2, 1, 7, 8, 6, 3, 4, 9, 5, 10, 11, 12, 13, 14, 18, 19, 16, 17, 15, 0]
        for i in range (20):
            mapping[mapping_count_sorted[i]] = mapping_order[i]

        ######################################################################################
        # SA algorithm
        Tmin = 0
        Tcurr = 30 # initial temperature
        prev_score = 99999
        # lq_num = 2 * len(self.unexecuted_gates)
        cc = 0
        test_circuit = []
        while True:
             
            if Tcurr == 30:
                test_mapping = copy.deepcopy(mapping)
                temp_test_mapping = copy.deepcopy(mapping)
            temp_test_mapping = copy.deepcopy(test_mapping)

            # swap
            if Tcurr != 30:
                coordinate = self.gencoordinates(0, 7)
                q1, q2 = next(coordinate)
                temp_test_mapping = self.swap(test_mapping, test_mapping[q1], test_mapping[q2])
                

            # creat a scaled-down version of the original circuit
            probability_of_gates = []

            # calculate the probability of each logic qubit  
            for i in range(len(gate_combination)):
                probability_of_gates.append(gate_appear_time_arr[gate_appear_time_sorted[i]] / len(self.unexecuted_gates))
            
            # creat a scaled-down circuit with 5 CNOT Gates
            if Tcurr == 30:
                for _ in range(5):
                    test_gate = self.generate_gate(probability_of_gates, gate_appear_time_sorted, gate_combination) 
                    test_circuit.append(test_gate)
            
            test = QubitRouting()
            test.unexecuted_gates = copy.deepcopy(test_circuit)
            #print(test.unexecuted_gates)
            #################################################################################################################################
            test_start = time.time()
            test_circuit_check = []
            for _ in range(V):
                test_circuit_check.append(0)
            
            while test.unexecuted_gates != []:

                if test.unexecuted_gates_front_layer == []:
                    test_circuit_check = []
                    for _ in range(V):
                        test_circuit_check.append(0)
                
                for i in range(len(test.unexecuted_gates)):
                    # print(test.unexecuted_gates)
                    if test_circuit_check[test.unexecuted_gates[i][0]] == 1 or test_circuit_check[test.unexecuted_gates[i][1]] == 1:
                        break
                    else:
                        test.unexecuted_gates_front_layer.append(test.unexecuted_gates[i])
                        test_circuit_check[test.unexecuted_gates[i][0]] = 1
                        test_circuit_check[test.unexecuted_gates[i][1]] = 1

                test_swap_candidates = test.generate_swap_candidate(test.graph)
                test_action = test.generate_action(test.graph, temp_test_mapping, test.unexecuted_gates_front_layer, test_swap_candidates)
                
                
                if test_action != []:
                    test.depth += 1
        
                test.operate_action(test.graph, temp_test_mapping, test_action, test.unexecuted_gates_front_layer)
                
            
            test_end = time.time()
            #################################################################################################################################
            
            # Evaluate equation
            score = test.swap_num * 3

            k = self.acceptance_probability(prev_score, score, Tcurr)
            
            if rd.random() < k:
                prev_score = score
                if Tcurr != 30:
                    self.swap2(test_mapping, test_mapping[q1], test_mapping[q2])
                print('tentative mapping: ', test_mapping)
                #cc += 1
                
            elif k < 0.0000000000000000000001:
                break
            #else:
                
            Tcurr -= 0.015
            cc += 1
            
        mapping = test_mapping
        print("\n(estimate times / search space) :", cc , "/" , math.factorial(7))
        print("mapping: ", mapping)

if __name__ == '__main__':
    q = QubitRouting()
    q.generate_mapping(mapping)
    
    start = time.time()
    circuit_check = []
    for _ in range(V):
        circuit_check.append(0)

    while q.unexecuted_gates != []:

        if q.unexecuted_gates_front_layer == []:
            circuit_check = []
            for _ in range(V):
                circuit_check.append(0)
        
        for i in range(len(q.unexecuted_gates)):
            if circuit_check[q.unexecuted_gates[i][0]] == 1 or circuit_check[q.unexecuted_gates[i][1]] == 1:
                break
            else:
                q.unexecuted_gates_front_layer.append(q.unexecuted_gates[i])
                circuit_check[q.unexecuted_gates[i][0]] = 1
                circuit_check[q.unexecuted_gates[i][1]] = 1

        swap_candidates = q.generate_swap_candidate(q.graph)
        action = q.generate_action(q.graph, mapping, q.unexecuted_gates_front_layer, swap_candidates)
        
        if action != []:
            q.depth += 1
       
        q.operate_action(q.graph, mapping, action, q.unexecuted_gates_front_layer)

    end = time.time()

    print("\n-----done-----")
    print("time(s):", end - start)
    print("g(add):", q.swap_num * 3)
    print("depth:", q.depth)
    
    