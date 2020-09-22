import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
import random as rnd
import ES

import numpy as np

n_hidden_neurons = 10


def init():
    """
    Initialise game environment
    """
    # create map for outputs
    experiment_name = 'testrun'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    controller = player_controller(n_hidden_neurons)

    env = Environment(experiment_name=experiment_name,
                      playermode='ai',
                      player_controller=controller,
                      )

    return env

def run_play(weights):
    stats = env.play(pcont=weights)
    fitness = stats[0]

    return fitness

def cross_rand(pop_data,best,population):
    """
    Performs random crossover
    """
    all_weights = []
    indeces = []

    # extract best individuals:
    if not best:
        indeces = list(pop_data.keys())
    else:
        indeces = list(pop_data.keys())[:best]

    # generate all new individuals from random crossover
    for i in range(population):
        # choose random individual from list:
        parent1, parent2 = np.random.choice(indeces, size=2, replace=False)

        parent1_data = pop_data[parent1][1]
        parent2_data = pop_data[parent2][1]

        # compose child of randomly selected genomes of parents
        child = []
        for i,c in enumerate(parent1_data):
            if rnd.random()>0.5:
                child += [parent1_data[i]]
            else:
                child += [parent2_data[i]]

        all_weights += [child]

    return np.array(all_weights)

def mutate(weights,gen):
    """
    adds normally distributed noise to the weights
    """
    mutated_weights = []
    for weight in weights:
        new_weight = np.random.normal(loc=weight, scale=0.5/(gen+1))
        if new_weight >= -1 and new_weight <= 1:
            mutated_weights.append(new_weight)
        elif new_weight < -1:
            mutated_weights.append(-1)
        else:
            mutated_weights.append(1)
    return np.array(mutated_weights)

def evo_alg(evo_type,env,n_hidden_neurons):
    """
    This function uses the two evolutionary algorithms
    to optimize the weights used in the neural network
    """
    # initialise parameters for NEAT1
    ngens = 10
    population = 20
    best = 10

    # initialise parameters for NEAT2
    if evo_type=='NEAT2':
        best = False

    pop_data = {}

    # initialize population randomly
    for ind in range(population):
        # initialise random weights

        weights = np.random.uniform(-1,1,size=(n_hidden_neurons+20*n_hidden_neurons+5+n_hidden_neurons*5))

        fitness = run_play(weights)
        pop_data[ind] = (fitness, weights)

    # perform evolutionary algorithm for all generations
    for gen in range(ngens):

        print(f'RUN: {gen+1}')

        # sort by fitness
        pop_data={k: v for k, v in sorted(pop_data.items(), key=lambda item: item[1][0], reverse=True)}

        # perform cross-over on best individuals
        all_weights = cross_rand(pop_data,best,population)

        # overwrite old population data
        pop_data = {}

        for ind in range(population):
            weights = mutate(all_weights[ind],gen)

            fitness = run_play(weights)
            pop_data[ind] = (fitness,weights)



if __name__ == '__main__':

    env = init()

    # run first algorithm
    # evo_alg('NEAT1', env, n_hidden_neurons)

    n_gens = 10
    n_pop  = 10
    k = 3
    l = int(2.5*n_pop)
    n_genes= n_hidden_neurons+20*n_hidden_neurons+5+n_hidden_neurons*5

    ES.evol_strat(n_hidden_neurons, n_gens, n_pop, env, n_genes, l, k)

    # run second algorithm
    # evo_alg('NEAT2', env, n_hidden_neurons)
