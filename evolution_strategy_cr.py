# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 19:58:07 2023

@author: mcamboim
"""

import numpy as np

# Comentar aqui
def getTwoDimensionalGaussian(std,cov):
    S = np.diag(std)
    R11 = getTwoDimensionalRotationMatrix(cov[0])
    R12 = getTwoDimensionalRotationMatrix(cov[1])
    T = R11 @ R12
    return T.T @ S.T @ np.random.normal(0,1,2)
    
# Comentar aqui
def getTwoDimensionalRotationMatrix(theta):
    R = np.zeros((2,2))
    R[0,0] = np.cos(theta)
    R[0,1] = np.sin(theta) * -1.0
    R[1,0] = np.sin(theta)
    R[1,1] = np.cos(theta)
    return R

def checkIfAngleIsBetween0And2Pi(theta):
    if(theta // (2*np.pi) > 0.0):
        theta = theta - 2*np.pi*(theta // (2*np.pi))
    return theta

def getEuclideanDistance(this_individual,other_individual):
    x_1,y_1 = this_individual[0],this_individual[1]
    x_2,y_2 = other_individual[0],other_individual[1]
    distance = np.sqrt((x_1 - x_2)**2 + (y_1 - y_2)**2)
    return distance
    
class EvolutionStrategyCR:
    
    def __init__(self):
        pass
    
    def popInit(self,pop_size):
        self.__pop = np.zeros((pop_size,6))
        self.__pop_fitness = np.zeros(pop_size)
        for individual_idx in range(pop_size):
            self.__pop[individual_idx,0:2] = np.random.uniform(-1,2,2)      # Solutions
            self.__pop[individual_idx,2:4] = np.random.uniform(0,0.5,2)     # Standard Deviation
            self.__pop[individual_idx,4:6] = np.random.uniform(0,2*np.pi,2) # Covariance Angle (0 ~ 2pi)
            
    def runEE(self,pop_size,var_std,cov_std,generations):
        # Begin Variables
        self.__pop_best_objective_by_generation = np.zeros(generations)
        self.__pop_mean_objective_by_generation = np.zeros(generations)        
        # 1. Initialization of the population
        self.popInit(pop_size)
        # 2. Get Fitness
        self.__pop_fitness = self.getPopFitness(self.population)
        # Run Genetic Algorithm
        for generation in range(generations):
            print(f'{generation+1}/{generations} -> {self.best_objective_function}')
            self.__pop_best_objective_by_generation[generation] = self.best_objective_function
            self.__pop_mean_objective_by_generation[generation] = self.mean_objective_function
            # Selection
            pop_selected = self.randomSelection(pop_size,pop_size*2)
            # Crossover
            pop_son = self.crossoverForRealVector(pop_selected)
            # Mutation
            pop_son = self.mutationForRealVector(pop_son,var_std,cov_std)
            pop_son_fitness = self.getPopFitness(pop_son)
            # Elitism
            self.elitism(pop_son, pop_son_fitness)
            self.__pop_fitness = self.getPopFitness(self.population)
            
    # Fitness ================================================================
    def getObjectiveFunction(self,individual):
        x = individual[0]
        y = individual[1]
        objective_function_value = x * np.sin(4*np.pi*x) - y * np.sin(4*np.pi*y+np.pi) + 1
        return objective_function_value
    
    def getPopFitness(self,pop):
        pop_size = pop.shape[0]
        pop_fitness = np.zeros(pop_size)
        for this_individual_idx in range(pop_size):
            objective_function_value = self.getObjectiveFunction(pop[this_individual_idx,:])
            pop_fitness[this_individual_idx] = objective_function_value
        return pop_fitness
    
    # Selection ==============================================================
    def randomSelection(self,pop_size,individuals_to_be_selected):
        pop_selected = np.zeros((individuals_to_be_selected,6))
        for pop_selected_idx in range(individuals_to_be_selected):
            if(pop_selected_idx % 2 == 0):
                selected_individual_idx = np.random.randint(0,pop_size)
                pop_selected[pop_selected_idx,:] = self.population[selected_individual_idx,:]
            else:
                last_individual = self.population[selected_individual_idx,:]
                pop_selected[pop_selected_idx,:] = self.speaction(selected_individual_idx,last_individual,phi_m,pop_size)
        return pop_selected
    
    def speaction(self,last_individual_idx,last_individual,phi_m,pop_size):
        individuals_that_can_be_selected_idx = list(range(pop_size))
        individuals_that_can_be_selected_idx.pop(last_individual_idx)
        np.random.shuffle(individuals_that_can_be_selected_idx)
        an_individual_was_selected = False
        for candidate_to_be_selected_idx in individuals_that_can_be_selected_idx:
            candidate = self.population[candidate_to_be_selected_idx,:]
            distance = getEuclideanDistance(last_individual, candidate)
            if distance < phi_m:
                an_individual_was_selected = True
                selected_individual = candidate
                break
        if not an_individual_was_selected:
            selected_individual_idx = np.random.randint(0,pop_size)
            selected_individual = self.population[selected_individual_idx,:]
        return selected_individual
    
    # Crossover ==============================================================
    def crossoverForRealVector(self,pop_selected):
        pop_son = np.copy(pop_selected)
        pop_son_size = pop_son.shape[0]
        # Now perform the crossover in the selected individuals
        for pop_crossover_idx in range(0,pop_son_size,2):
            individual_to_crossover_idx1 = pop_crossover_idx
            individual_to_crossover_idx2 = pop_crossover_idx+1
            individual_to_crossover_1 = np.copy(pop_selected[individual_to_crossover_idx1,:])
            individual_to_crossover_2 = np.copy(pop_selected[individual_to_crossover_idx2,:])
            
            son_1,son_2 = self.crossoverConvex(individual_to_crossover_1, individual_to_crossover_2)
            pop_son[individual_to_crossover_idx1,:] = np.copy(son_1)
            pop_son[individual_to_crossover_idx2,:] = np.copy(son_2)
        
        return pop_son
    
    def crossoverConvex(self,individual_to_crossover_1,individual_to_crossover_2):
        son_1 = np.zeros(6)
        son_2 = np.zeros(6)
        
        # Solution
        alpha = np.random.uniform(0,1)
        son_1[0:2] = alpha * individual_to_crossover_1[0:2] + (1.0-alpha) * individual_to_crossover_2[0:2]
        son_2[0:2] = alpha * individual_to_crossover_2[0:2] + (1.0-alpha) * individual_to_crossover_1[0:2]
        # STD and Covariance
        beta = np.random.uniform(0,1)
        son_1[2:6] = beta * individual_to_crossover_1[2:6] + (1.0-beta) * individual_to_crossover_2[2:6]
        son_2[2:6] = beta * individual_to_crossover_2[2:6] + (1.0-beta) * individual_to_crossover_1[2:6]
        
        return son_1,son_2        
    
    # Mutation ===============================================================
    def mutationForRealVector(self,pop_son,var_std,cov_std):
        for pop_son_idx in range(pop_son.shape[0]):
            # Mutation of the STD Values
            pop_son[pop_son_idx,2:4] = pop_son[pop_son_idx,2:4] * np.exp(np.random.uniform(0,var_std,2))
            pop_son[pop_son_idx,4:6] = pop_son[pop_son_idx,4:6] + np.random.uniform(0,cov_std,2) 
            # Check if Theta is between 0 and 2*pi
            pop_son[pop_son_idx,4] = checkIfAngleIsBetween0And2Pi(pop_son[pop_son_idx,4])
            pop_son[pop_son_idx,5] = checkIfAngleIsBetween0And2Pi(pop_son[pop_son_idx,5])
            # Mutation of solution values
            std = pop_son[pop_son_idx,2:4]
            cov = pop_son[pop_son_idx,4:6]
            pop_son_mut = pop_son[pop_son_idx,0:2] + getTwoDimensionalGaussian(std,cov)
            while((pop_son_mut[0] < -1.0 or pop_son_mut[0] > 2.0) or (pop_son_mut[1] < -1.0 or pop_son_mut[1] > 2.0)):
                pop_son_mut = pop_son[pop_son_idx,0:2] + getTwoDimensionalGaussian(std,cov)
            pop_son[pop_son_idx,0:2] = np.copy(pop_son_mut)
                
        return pop_son
    
    # Elitism ================================================================
    def elitism(self,pop_son,pop_son_fitness):
        pop_size = self.population.shape[0]
        all_individuals = np.zeros((len(pop_son)+pop_size,6))
        all_individuals_fitness = np.zeros(len(pop_son)+pop_size)
        all_individuals[:pop_size,:] = np.copy(self.population)
        all_individuals[pop_size:,:] = np.copy(pop_son)
        all_individuals_fitness[:pop_size] = np.copy(self.fitness)
        all_individuals_fitness[pop_size:] = np.copy(pop_son_fitness)
        ordered_all_individuals_idx = np.flip(np.argsort(all_individuals_fitness))
        self.__pop = np.copy(all_individuals[ordered_all_individuals_idx[:pop_size]])
    
    # Properties =============================================================
    @property
    def population(self):
        return self.__pop
    
    @property
    def fitness(self):
        return self.__pop_fitness
    
    @property
    def best_fitness(self):
        return np.max(self.__pop_fitness)
    
    @property
    def best_objective_through_generations(self):
        return self.__pop_best_objective_by_generation
    
    @property
    def mean_objective_through_generations(self):
        return self.__pop_mean_objective_by_generation
    
    @property
    def best_objective_function(self):
        best_fitness_idx = np.argmax(self.fitness)
        best_fitness_objetctive_function = self.getObjectiveFunction(self.population[best_fitness_idx,:])
        return best_fitness_objetctive_function
    
    @property
    def mean_objective_function(self):
        pop_size = self.population.shape[0]
        fitnesses = np.zeros(pop_size)
        for individual_idx in range(pop_size):
            fitnesses[individual_idx] = self.getObjectiveFunction(self.population[individual_idx,:])
        return np.mean(fitnesses)
    
        
            