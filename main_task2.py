import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
import random as rnd
import numpy as np

from DE import differential_evolution

n_hidden_neurons = 10

def init(enemies, multiplemode):
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
                      enemies=enemies,
                      multiplemode=multiplemode
                      )

    return env

if __name__ == '__main__':



    n_runs = 5
    n_gens = 3
    n_pop  = 10

    # enemies = [2,1,2,1]
    enemies = [6,7,4]
    indeces = np.arange(0, n_runs*len(enemies))
    print(indeces)

    F = 1.5
    CR = 0.8

    n_genes= n_hidden_neurons+20*n_hidden_neurons+5+n_hidden_neurons*5

    run = 1

    multiplemode = 'no'

    env = init([enemies[0]], multiplemode)

    pop_data = False

    best_solutions = {i: [] for i in enemies}
    print(best_solutions)

    for i in indeces:
        enemy = enemies[i%len(enemies)]
        print("\n"+f"Iteration {i} for enemy {enemy}."+"\n")

        pop_data, best_ind = differential_evolution(n_hidden_neurons, n_gens, n_pop, env, n_genes, run,
                               enemy, F, CR, pop_data)

        best_solutions[enemy] += [best_ind]
        print(best_solutions)

    all_enemies = {}
    gain_measure = 0
    for i in range(1,9):
        env.update_parameter('enemies', [i])
        fitness = env.play(pcont=best_solutions[enemies[-1]][-1][1])
        all_enemies[i] = fitness[0]
