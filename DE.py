import numpy as np
import random as rnd

# from scipy.optimize import differential_evolution as diff_evol

"""
This file implements the Differential Evolution algorithm using scipy
"""


def get_best_ind(pop_data, best_ind):
    """
    Checks if a new best individual is found
    """
    for i in pop_data:
        if i['fitness'] > best_ind[0]:
            best_ind = (i['fitness'], i['target'])

    return best_ind


def run_play(weights, env):
    """
    Runs play function in environment, returns fitness for given weights
    """
    stats = env.play(pcont=weights)
    fitness = stats[0]

    return fitness


def recalc_fitness(pop_data, env):
    """
    Recalculates fitness for new enemy
    """
    for ind in pop_data:
        fitness = run_play(ind['target'], env)
        print(fitness)
        pop_data[pop_data.index(ind)]['fitness'] = fitness

    best_ind = (0, [])

    best_ind = get_best_ind(pop_data, best_ind)

    return pop_data, best_ind


def random_weights(env, n_genes, n_pop):
    """
    Gives initial values for random weights
    """

    pop_data = []
    best_ind = (0,[])

    for ind in range(n_pop):

        # initialise random weights
        weights = np.random.uniform(-1,1,size=n_genes)

        fitness = run_play(weights, env)
        pop_data += [{'fitness': fitness, 'target': weights}]

    best_ind = get_best_ind(pop_data, best_ind)

    return pop_data, best_ind


def mutate(pop_data, n_genes, F):
    """
    Mutates weights, creates mutation vectors.
    """

    # create mutation vector for each individual
    for ind in pop_data:
        mut_vect = []

        # for each gene in genome, choose new gene
        for gene in range(n_genes):
            # randomly choose 3 parents, no replacement
            targets = np.random.choice(pop_data, size=3, replace=False)

            t1 = targets[0]['target']
            t2 = targets[1]['target']
            t3 = targets[2]['target']

            # add mutated gene
            mut_vect += [t1[gene] + F*(t2[gene]-t3[gene])]

        pop_data[pop_data.index(ind)]['mut_vect'] = np.array(mut_vect)

    return pop_data


def crossover(pop_data, n_genes, CR):
    """
    Performs crossover, creates trial vectors
    """

    # create trial vector for each individual
    for ind in pop_data:
        target = ind['target']
        mut_vect = ind['mut_vect']

        trial = []

        # randomly choose index of mutation vector that will definitely be
        # put in the trial vector, so trial vector can never be equal to the
        # target vector
        mut_index = np.random.choice(range(n_genes))

        # choose gene from trial or target vector, based on chance.
        for gene in range(n_genes):
            if np.random.uniform() <= CR or gene == mut_index:
                trial += [mut_vect[gene]]
            else:
                trial += [target[gene]]

        pop_data[pop_data.index(ind)]['trial'] = np.array(trial)

    return pop_data


def selection(env, pop_data, best_ind):

    new_pop_data = []

    for ind in pop_data:

        trial = ind['trial']

        trial_fitness = run_play(trial, env)

        target_fitness = ind['fitness']

        print(f'FITNESS TARGET: {target_fitness}')
        print(f'FITNESS TRIAL: {trial_fitness}')

        if trial_fitness > ind['fitness']:
            print('Choose TRIAL')
            new_pop_data += [{'fitness': trial_fitness, 'target': trial}]
        else:
            new_pop_data += [{'fitness': ind['fitness'], 'target': ind['target']}]
            print('Choose TARGET')

    best_ind = get_best_ind(pop_data, best_ind)

    return new_pop_data, best_ind


def differential_evolution(n_hidden_neurons, n_gens, n_pop, env, n_genes,
                           run, enemy, F, CR, pop_data):

    env.update_parameter('enemies',[enemy])

    if not pop_data:
        pop_data, best_ind = random_weights(env, n_genes, n_pop)
    else:
        pop_data, best_ind = recalc_fitness(pop_data, env)


    # loop over generations
    for gen in range(n_gens):
        print(f'\n GENERATION: {gen} \n')

        pop_data = mutate(pop_data, n_genes, F)

        pop_data = crossover(pop_data, n_genes, CR)

        pop_data, best_ind = selection(env, pop_data, best_ind)

    return pop_data, best_ind
