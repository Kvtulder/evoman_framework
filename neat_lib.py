import sys
import os
import neat
import pickle

sys.path.insert(0, 'evoman')
from environment import Environment


class Controller:
    def __init__(self, config):
        self.config = config

   
    def control(self, inputs, genome):
        # normalise input, taken from demo_controller
        inputs = (inputs - min(inputs)) / float((max(inputs) - min(inputs)))
        net = neat.nn.FeedForwardNetwork.create(genome, self.config)
        outputs = net.activate(inputs)

        # create output array
        action_array = []
        for output in outputs:
            action_array.append(int(output > 0.5))

        return action_array


def fitness_func(genomes, config):

    for genome_id, genome in genomes:
        stats = env.play(pcont=genome)
        print("asdasd12", stats[0])
        genome.fitness = stats[0]
       


def neat_run(config, population, generations, mode, run, gen, enemies, enemy):



    # Create the population, which is the top-level object for a NEAT run.
    best_solution = population.run(fitness_func, generations)

    winner_net = neat.nn.FeedForwardNetwork.create(best_solution, config)
    print(winner_net)
    print(list(population.population.items()))
    save_gen(mode, run, gen, list(population.population.items()), enemies, enemy)
    return population, best_solution

def save_gen(mode, run, gen, pop_data, enemies, enemy):
    """
    Saves run data to csv
    """
    map_name = f'{mode}_enemies_{str(enemies)}'
    if not os.path.exists(map_name):
        os.makedirs(map_name)

    string = str(enemy) + ',' + str(pop_data[0][1].fitness)
    for i in range(1, len(pop_data)):
        string += ", "
        if pop_data[i][1].fitness == None:
            pop_data[i][1].fitness = env.play(pcont=pop_data[i][1])[0]
        string += str(pop_data[i][1].fitness)
    f = open(map_name + '/' + mode + '_' + str(enemies)+ '_' + str(run) + ".csv", "a")
    f.write(string + '\n')
    f.close()


def get_best_ind(env, pop_data):
    """
    Test which weights perform best on all enemies:
    """

    fitness_all_enemies = {}
    for i in range(len(pop_data)):
        print(f'\n Check performance for individual {i}:')
        fitness_all_enemies[i] = []
        for enemy in range(1,9):
            env.update_parameter('enemies', [enemy])
            fitness = env.play(pcont=pop_data[i][1])
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
    # print(best_ind)
    save_bestind(mode, run, enemies, pop_data[best_ind_index][1], avg_best_fitness)

def save_bestind(mode, run, enemies, best_ind, avg_best_fitness):
    """
    Saves best individual data of each run
    """
    map_name = f'{mode}_enemies_{str(enemies)}'
 
    with open(map_name + '/' + mode + str(run) +  '_best_' + str(enemies) + "_avg_" + str(avg_best_fitness), "wb") as file:
        pickle.dump(best_ind, file)
    

if __name__ == '__main__':

    



    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')

    # set parameters for game
    # enemies = [6,7] # set of enemies the weights are trained on
    enemies = [6, 7, 4]
    n_runs    = 10    # amount the total evolution is performed, standard is 10
    n_subgens = 3    # number of generations trained on each enemy before passed
                     # to the other enemies
    n_gens    = 5*len(enemies)   # number of generations for which the whole set of enemies
                     # is trained
    mode = "NEAT"

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # create map for outputs
    experiment_name = 'testrun'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)


    pop_data = neat.Population(config)

    # evolution
    for run in range(n_runs):

        for gen in range(n_gens):
            # select which enemy to play against
            enemy = enemies[gen%len(enemies)]
            env = Environment(experiment_name=experiment_name, speed="fastest", enemies=[enemy], playermode='ai', player_controller = Controller(config))

            print("\n"+f"Iteration {gen} for enemy {enemy}."+"\n")
            pop_data, best_solution = neat_run(config, pop_data, n_subgens, mode, run, gen, enemies, enemy)
            

        # get best individual and save to csv
        get_best_ind(env, list(pop_data.population.items()))
