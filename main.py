# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 19:45:26 2023

@author: mcamboim
"""
from evolution_strategy import EvolutionStrategy
from evolution_strategy_fs import EvolutionStrategyFS
from evolution_strategy_fs_cr import EvolutionStrategyFSCR
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')
plt.rcParams['axes.linewidth'] = 2.0
plt.rc('axes', axisbelow=True)
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}
plt.rc('font', **font)

pop_size = 100
var_std = 0.1
cov_std = np.pi/10.0
phi_s = 0.5
phi_m = 0.5
generations = 202
executions = 4
EE_type = 2 # 0 -> Simple | 1 -> Fitness Share | 2 -> Fitness Share + Speciation

if EE_type == 0:
    ee_alg = EvolutionStrategy()
elif EE_type == 1:
    ee_alg = EvolutionStrategyFS()
else:
    ee_alg = EvolutionStrategyFSCR()
best_objective = np.zeros((generations,executions))
mean_objective = np.zeros((generations,executions))

for execution_idx in range(executions):
    print(f'\nExecução {execution_idx+1}/{executions}')
    if(EE_type == 0):
        ee_alg.runEE(pop_size=pop_size,var_std=var_std,cov_std=cov_std,generations=generations)
    elif(EE_type == 1):
        ee_alg.runEE(pop_size=pop_size,var_std=var_std,cov_std=cov_std,phi_s=phi_s,generations=generations)
    else:
        ee_alg.runEE(pop_size=pop_size,var_std=var_std,cov_std=cov_std,phi_s=phi_s,phi_m=phi_m,generations=generations)
    best_objective[:,execution_idx] = ee_alg.best_objective_through_generations
    mean_objective[:,execution_idx] = ee_alg.mean_objective_through_generations
    
# Figures
best_objective_idx = np.argmax(best_objective[-1,:])
plt.figure(figsize=(12,6),dpi=150)
plt.plot(best_objective[:,best_objective_idx],lw=2,c='b')
plt.ylabel('Função Objetivo []')
plt.xlabel('# de Gerações [n]')
plt.legend(['Evolução da Função Objetivo','Valor de Referência'])
plt.xlim([0,generations-1])
plt.grid(True,ls='dotted')
plt.tight_layout()

# level curve
points = 50
x1_vec = np.zeros((points,points))
x2_vec = np.zeros((points,points))
y_vec = np.zeros((points,points))
x1_idx = 0
x2_idx = 0
for x1 in np.linspace(-1.0,2.0,points):
    for x2 in np.linspace(-1.0,2.0,points):
        x1_vec[x1_idx,x2_idx] = x1
        x2_vec[x1_idx,x2_idx] = x2
        y_vec[x1_idx,x2_idx] = ee_alg.getObjectiveFunction([x1,x2])
        x2_idx = x2_idx + 1
    x2_idx = 0
    x1_idx = x1_idx + 1
                                         
plt.figure(figsize=(12,6),dpi=150)
plt.contour(x1_vec, x2_vec, y_vec,levels=np.linspace(-3,4.5,500),cmap='coolwarm',zorder=0)
plt.xlim([-1,2])
plt.ylim([-1,2])
for invidivual_idx in range(pop_size):
    plt.scatter(ee_alg.population[invidivual_idx,0],ee_alg.population[invidivual_idx,1],zorder=invidivual_idx+1,s=25,edgecolor='k',c='b')

# Fitness Evolution
plt.figure(figsize=(12,6),dpi=150)
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(best_objective[:,i],lw=2,c='b')
    plt.plot(mean_objective[:,i],lw=2,c='r')
    #plt.plot([17_212_548] * generations,ls='--',lw=2,c='r')
    plt.xlim([0,generations-1])
    plt.grid(True,ls='dotted')
    #
    #plt.xlabel('# de Gerações [n]')
    if(i==0 or i == 2):
        plt.ylabel('$\phi(n)$')
    if(i>=2):
        plt.xlabel('# de Geracoes [n]')
    if(i==0):
        plt.legend(['Melhor','Media'])
plt.tight_layout()



