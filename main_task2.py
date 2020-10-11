import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
import random as rnd
import numpy as np

from DE import differential_evolution

n_hidden_neurons = 10


def init(enemies):
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
                      enemies=enemies
                      )

    return env


def save_bestind(mode, run, enemies, best_ind, avg_best_fitness):
    """
    Saves best individual data of each run
    """
    map_name = f'{mode}_enemies_{str(enemies)}'

    string = str(avg_best_fitness)
    for weight in best_ind:
        string += ", "
        string += str(weight)
    f = open(map_name + '/' + mode + '_best_' + str(enemies) + ".csv", "a")
    f.write(string + '\n')
    f.close()


def get_best_ind(env, pop_data):
    """
    Test which weights perform best on all enemies:
    """

    fitness_all_enemies = {}
    for i, ind in enumerate(pop_data):
        # print(type(pop_data))
        # print(type(ind))
        print(f'\n Check performance for individual {i}:')
        fitness_all_enemies[i] = []
        for enemy in range(1,9):
            env.update_parameter('enemies', [enemy])
            fitness = env.play(pcont=ind['target'])
            fitness_all_enemies[i] += [fitness[0]]
        print(fitness_all_enemies)

    # choose best individual and save
    # print(fitness_all_enemies)
    for i in fitness_all_enemies:
        fitness_all_enemies[i] = sum(fitness_all_enemies[i])/len(fitness_all_enemies[i])
    fitness_all_enemies = {k: v for k, v in sorted(fitness_all_enemies.items(), key=lambda item: item[1])}
    # print(fitness_all_enemies)

    # save best individuals
    # print(list(fitness_all_enemies.keys())[-1])
    best_ind_index = list(fitness_all_enemies.keys())[-1]
    avg_best_fitness = fitness_all_enemies[best_ind_index]
    best_ind = pop_data[best_ind_index]['target']
    # print(best_ind)
    save_bestind(mode, run, enemies, best_ind, avg_best_fitness)


if __name__ == '__main__':

    # set parameters for game
    # enemies = [6,7] # set of enemies the weights are trained on
    enemies = [6,7,4]
    n_runs    = 1    # amount the total evolution is performed, standard is 10
    n_subgens = 3    # number of generations trained on each enemy before passed
                     # to the other enemies
    n_gens    = 5*len(enemies)   # number of generations for which the whole set of enemies
                     # is trained
    n_pop     = 15   # number of individuals in each population
    mode      = 'DE' # game mode

    F = 0.3
    CR = 0.3

    n_genes= n_hidden_neurons+20*n_hidden_neurons+5+n_hidden_neurons*5

    env = init([enemies[0]])

    # evolution
    pop_data = False
    for run in range(n_runs):

        for gen in range(n_gens):
            # select which enemy to play against
            enemy = enemies[gen%len(enemies)]
            print("\n"+f"Iteration {gen} for enemy {enemy}."+"\n")

            # perform evolution for enemy, n_gens times, get evolved weights
            pop_data = differential_evolution(mode, n_hidden_neurons, n_subgens, n_pop,
                                              env, n_genes, run, enemy, enemies, F,
                                              CR, pop_data)

        # get best individual and save to csv
        get_best_ind(env, pop_data)
