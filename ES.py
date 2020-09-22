"""
This file implements a Evolutionary Strategies Algorithm.
"""

import numpy as np
import random as rnd
import pandas as pd
import sys, os

def avg(list):
    return sum(list)/len(list)

def run_play(weights, env):
    stats = env.play(pcont=weights)
    fitness = stats[0]

    return fitness


def global_recomb(pop_data, n_pop, n_genes, l):
    """
    Performs global recombination. Returns new weights.
    Uses (mu, lambda), thus disgarding all parents.
    """

    new_weights = []

    for i in range(l):
        child = []

        for j in range(n_genes):
            parent = np.random.choice(range(len(pop_data)))
            # print(parent)
            child += [pop_data[parent][1][j]]

        new_weights += [child]

    return np.array(new_weights)


def succes_rate(old_fitness, new_fitness):
    """
    Computes succes rate
    """
    succ = 0
    old_fitness = sorted(old_fitness,reverse=True)

    for i in range(len(new_fitness)):
        if new_fitness[i] > old_fitness[i]:
            succ += 1

    return succ/len(new_fitness)


def mutation(sr, new_weights):
    """
    Performs mutation
    """
    c = 0.8
    sigma = 1

    if sr > 0.2:
        sigma = sigma/c
    elif sr < 0.2:
        sigma = sigma*c

    for i in range(len(new_weights)):
        for j,gen in enumerate(new_weights[i]):
            new_weights[i][j] = gen + np.random.normal(0,sigma)

    return new_weights



def save_run(run, pop_data, mode, enemy):
    """
    Saves run data to csv
    """
    map_name = mode + '_' + str(enemy)
    if not os.path.exists(map_name):
        os.makedirs(map_name)

    string = str(pop_data[0][0])
    for i in range(1, len(pop_data)):
        string += ", "
        string += str(pop_data[i][0])
    f = open(map_name + '/' + mode + '_' + str(enemy) + '_' + str(run) + ".csv", "a")
    f.write(string + '\n')
    f.close()


def save_bestind(run, mode, enemy, best_ind):
    """
    Saves best individual data of each run
    """
    map_name = mode + '_' + str(enemy)

    string = str(best_ind[0]) + str(best_ind[1][0])
    for i in range(1, len(best_ind[1])):
        string += ", "
        string += str(best_ind[1][i])
    f = open(map_name + '/' + mode + '_best_' + str(enemy) + ".csv", "a")
    f.write(string + '\n')
    f.close()

def evol_strat(mode, n_hidden_neurons, n_gens, n_pop, env, n_genes, l, k, run, enemy):
    """
    Main function, runs Evolutionary Strategies algorithm
    """

    pop_data = []
    best_ind = (0,[])

    for ind in range(l):

        # initialise random weights
        weights = np.random.uniform(-1,1,size=n_genes)

        fitness = run_play(weights, env)
        pop_data += [(fitness, weights)]

    for i in pop_data:
        if i[0]>best_ind[0]:
            best_ind = i

    sr = [0.2]
    overall_sr = sr[0]

    save_run(0, pop_data, mode, enemy)

    for gen in range(n_gens):
        print()
        print(f'GENERATION: {gen+1}')
        print()

        # sort population not needed as ES uses uniform parent selection

        # perform global recombination and mutation, based on last succes_rate
        new_weights = global_recomb(pop_data, n_pop, n_genes, l)
        new_weights = mutation(overall_sr, new_weights)

        old_fitness = [i[0] for i in pop_data]
        new_fitness = []

        for ind in range(l):
            new_fitness += [run_play(new_weights[ind], env)]

        # if (mu, lambda), throw away parents
        if mode == '(mu, lambda)':
            pop_data = []

        # put new weights in population data
        for i,c in enumerate(new_fitness):
            pop_data += [(c, new_weights[i])]

        print(f'LEN POP_DATA: {len(pop_data)}')

        # select best performing indivisuals
        pop_data = sorted(pop_data, key=lambda tup: tup[0], reverse=True)
        pop_data = pop_data[:n_pop]

        print(f'LEN POP_DATA: {len(pop_data)}')

        new_fitness = [i[0] for i in pop_data]

        # compute succes rate for each round. update overal sr each k rounds
        sr += [succes_rate(old_fitness, new_fitness)]

        if gen%k==0:
            overall_sr = avg(sr[-k:])
            print(f'overall succes rate: {overall_sr}')

        print(f'SUCCES RATE: {sr}')

        for i in pop_data:
            if i[0]>best_ind[0]:
                best_ind = i

        # print(best_ind)

        save_run(run+1, pop_data, mode, enemy)

    # write best result to csv
    save_bestind(run, mode, enemy, best_ind)
