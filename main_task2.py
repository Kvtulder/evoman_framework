import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
import random as rnd
import numpy as np

from DE import differential_evolution

n_hidden_neurons = 10

def init(enemy):
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
                      enemies=[enemy]
                      )

    return env

if __name__ == '__main__':



    n_runs = 10
    n_gens = 10
    n_pop  = 10

    enemies = [2,1,2,1]

    F = 1.5
    CR = 0.8

    n_genes= n_hidden_neurons+20*n_hidden_neurons+5+n_hidden_neurons*5

    run = 1

    pop_data = False
    for enemy in enemies:
        env = init(enemy)
        pop_data = differential_evolution(n_hidden_neurons, n_gens, n_pop, env, n_genes, run,
                               enemies, F, CR, pop_data)
