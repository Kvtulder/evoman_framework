import numpy as np
import random as rnd
import sys, os

"""
This file implements the Differential Evolution algorithm using scipy
"""


def save_gen(mode, run, gen, pop_data, enemies, enemy):
    """
    Saves run data to csv
    """
    map_name = f'{mode}_enemies_{str(enemies)}'
    # enemies_name = ''
    # for i in enemies:
    #     enemies_name += (str(i) + '_')
    # map_name += enemies_name

    if not os.path.exists(map_name):
        os.makedirs(map_name)

    string = str(enemy) + ',' + str(pop_data[0]['fitness'])
    for i in range(1, len(pop_data)):
        string += ", "
        string += str(pop_data[i]['fitness'])
    f = open(map_name + '/' + mode + '_' + str(enemies)+ '_' + str(run) + ".csv", "a")
    f.write(string + '\n')
    f.close()


def save_bestind(mode, run, enemies, best_ind, avg_best_fitness):
    """
    Saves best individual data of each run
    """
    map_name = f'{mode}_enemies_{str(enemies)}'
    # enemies_name = ''
    # for i in enemies:
    #     enemies_name += (str(i) + '_')
    # map_name += enemies_name

    string = str(avg_best_fitness)
    for weight in best_ind:
        string += ", "
        string += str(weight)
    f = open(map_name + '/' + mode + '_best_' + str(enemies) + ".csv", "a")
    f.write(string + '\n')
    f.close()


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
        pop_data[pop_data.index(ind)]['fitness'] = fitness

    # best_ind = (0, [])

    # best_ind = get_best_ind(pop_data, best_ind)

    return pop_data


def random_weights(env, n_genes, n_pop):
    """
    Gives initial values for random weights
    """

    pop_data = []
    # best_ind = (0,[])

    for ind in range(n_pop):

        # initialise random weights
        weights = np.random.uniform(-1,1,size=n_genes)

        fitness = run_play(weights, env)
        pop_data += [{'fitness': fitness, 'target': weights}]

    # best_ind = get_best_ind(pop_data, best_ind)

    return pop_data


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


def selection(env, pop_data):
    """
    Evaluates trial vector, selects best from target and trial vector
    """

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

    # best_ind = get_best_ind(pop_data, best_ind)

    return new_pop_data


def differential_evolution(mode, n_hidden_neurons, n_gens, n_pop, env, n_genes,
                           run, enemy, enemies, F, CR, pop_data):
    """
    Performs differential evolution
    """

    env.update_parameter('enemies',[enemy])

    if not pop_data:
        pop_data = random_weights(env, n_genes, n_pop)
    else:
        pop_data = recalc_fitness(pop_data, env)


    # loop over generations
    for gen in range(n_gens):
        print(f'\n GENERATION: {gen} \n')

        pop_data = mutate(pop_data, n_genes, F)

        pop_data = crossover(pop_data, n_genes, CR)

        pop_data = selection(env, pop_data)

        save_gen(mode, run, gen, pop_data, enemies, enemy)

    return pop_data
