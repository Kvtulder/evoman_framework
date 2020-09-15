import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

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

    env = Environment(experiment_name=experiment_name,
                      playermode='ai',
                      player_controller=player_controller(n_hidden_neurons),
                      )

    return env


if __name__ == '__main__':

    env = init()

    # use first EA
    # TODO:

    # use second EA
    # TODO:
    weights = np.random.rand(10)
    weights = np.append(weights,[np.random.rand(20,n_hidden_neurons)])
    weights = np.append(weights,[np.random.rand(5)])
    weights = np.append(weights,[np.random.rand(10,5)])
    print(weights)
    env.play(pcont=weights)
