from random import seed
import numpy as np
import random

fit = 0
M = 20 # rows (genome/individual length) 
N = 10 # columns (population size/# of individuals)

# Create an array full random numbers
# Threshold it to produce random true/false values
pop = np.random.rand(N,M) > .5

# build an empty array
arr = [[0 for i in range(M)] for j in range(N)]

# converting boolean array to 0s and 1s
for i in range(N):
    for j in range(M):
        if pop[i][j] == True:
            arr[i][j] = 1
        else:
            arr[i][j] = 0
pop = arr

# produces an offspring from 2 parents
def crossover(individual1,individual2): 
    offspring = np.concatenate((individual1[0:10],individual2[10:20]))

    # 10% chance of mutation
    if np.random.rand() <= 0.1:
        # flipping # in 1 random index
        index = np.random.choice(M,1,replace = False)
        if offspring[index] == 1:
            offspring[index] = 0
        else:
            offspring[index] = 1
        #print("mutation: " + str(offspring))

    return offspring
    
# finds a mate 
def tournament(pop):
    # randomly selects 3 individuals from the population
    index = np.random.choice(len(pop),3,replace = False)
    subset = []
    for i in index:
        subset.append(pop[i])
    
    # calculating fitness on the smaller sample
    fitness = np.sum(arr, axis = 1)

    # finding best mate from subset
    mate = [] 
    highest = 0
    for i in range(len(fitness)):
        if highest < fitness[i]:
            highest = fitness[i]
            mate = arr[i]
    
    return mate
    
# selects best individual from current population
def elitism(pop):
    global fit
    fitness = np.sum(pop, axis = 1)
    best = []
    highest = 0 
    for i in range(len(pop)-1):
        if highest < fitness[i]:
            highest = fitness[i]
            best = pop[i]
    fit = highest
    return best

epoch = 1 
# while not converged (while the perfect individual is not found yet)
while(fit != M):  
    print("Epoch " + str(epoch))
    print("individuals:")
    print(np.array(pop))
    new_pop = []
    while len(new_pop) < len(pop):
        for i in range(len(pop)):
            # 50% chance to crossover
            if np.random.rand() <= 0.5:    
                parent1 = pop[i]
                temp = np.delete(pop,i,0) # new mate must not be same as parent1
                parent2 = tournament(temp)
                
                # calculating crossover
                offspring1 = crossover(parent1,parent2)
                offspring2 = crossover(parent2,parent1) 
                new_pop.append(offspring1)
                new_pop.append(offspring2)
    
    # calculating fitness
    fitness = np.sum(pop, axis = 1)
    for j in fitness:
        if j == M:
            print("perfect individual with fitness value of " + str(M) + " found in this generation")
            break
    
    # performing elistism    
    best_ind = elitism(pop)
    new_pop.append(best_ind)
    print("best individual (by elitism) is " + str(best_ind) + " with fitness value " + str(fit))  
    print()
        
    # replace pop with offsprings 
    pop = new_pop
    epoch += 1

# did the population converge?
if fit == M:
    print("The perfect individual with a fitness of " + str(fit) + "is produced")
    
    
     



