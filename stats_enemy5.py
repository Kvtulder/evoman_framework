import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D

# which generations to plot
gens = [i for i in range(16)]

modes = ['(mu, lambda)', '(mu+lambda)']
enemies = [1, 2, 5]
all_fitness = {}
means = {}
means_std = {}
best_fit = {}
best_fit_std = {}

# which enemy fitness to plot
ENEM = 1
fontsize=15

#mu, lambda
for enemy in enemies:
    path = rf'C:\Users\guido\master CLS\evolutionary computing\evoman_framework\EC resultaten\(mu, lambda)_{enemy}'
    dir_mulambda = path

    # run: mean fitness for each run
    fitness = {}
    all_best_fit = {}
    for filename in os.listdir(dir_mulambda):
        fit = pd.read_csv(path + '/' + filename, header=None)
        for i in gens:
            if i+1 in fitness:
                fitness[i+1] += [np.mean(fit.iloc[i, :])]
            else:
                fitness[i+1] = [np.mean(fit.iloc[i, :])]

            if i+1 in all_best_fit:
                all_best_fit[i+1] += [np.max(fit.iloc[i, :])]
            else:
                all_best_fit[i+1] = [np.max(fit.iloc[i, :])]

    all_fitness[enemy] = fitness
    means[enemy] = np.array([np.mean(value) for key, value in fitness.items()])
    means_std[enemy] = np.array([np.std(value) for key, value in fitness.items()])
    best_fit[enemy] = np.array([np.mean(value) for key, value in all_best_fit.items()])
    best_fit_std[enemy] = np.array([np.std(value) for key, value in all_best_fit.items()])


# plotting of the mu, lambda method
plt.plot(gens, best_fit[ENEM], '-.', color='black')
plt.fill_between(gens, best_fit[ENEM]-best_fit_std[ENEM], best_fit[ENEM]+best_fit_std[ENEM], color='grey', alpha=0.5)
plt.title(f'Enemy {ENEM}', fontsize= fontsize)
plt.plot(gens, means[ENEM], color='black')
plt.fill_between(gens, means[ENEM]-means_std[ENEM], means[ENEM]+means_std[ENEM], color='grey', alpha=0.5)
plt.xlim(0, max(gens))

all_fitness = {}
means = {}
means_std = {}
best_fit = {}
best_fit_std = {}

for enemy in enemies:
    path = rf'C:\Users\guido\master CLS\evolutionary computing\evoman_framework\EC resultaten\(mu+lambda)_{enemy}'
    dir_mulambda = path

    # run: mean fitness for each run
    fitness = {}
    all_best_fit = {}
    for filename in os.listdir(dir_mulambda):
        fit = pd.read_csv(path + '/' + filename, header=None)
        for i in gens:
            if i+1 in fitness:
                fitness[i+1] += [np.mean(fit.iloc[i, :])]
            else:
                fitness[i+1] = [np.mean(fit.iloc[i, :])]

            if i+1 in all_best_fit:
                all_best_fit[i+1] += [np.max(fit.iloc[i, :])]
            else:
                all_best_fit[i+1] = [np.max(fit.iloc[i, :])]

    all_fitness[enemy] = fitness
    means[enemy] = np.array([np.mean(value) for key, value in fitness.items()])
    means_std[enemy] = np.array([np.std(value) for key, value in fitness.items()])
    best_fit[enemy] = np.array([np.mean(value) for key, value in all_best_fit.items()])
    best_fit_std[enemy] = np.array([np.std(value) for key, value in all_best_fit.items()])

# plotting of the mu + lambda method
plt.plot(gens, best_fit[ENEM], '-.', color='darkblue')
plt.fill_between(gens, best_fit[ENEM]-best_fit_std[ENEM], best_fit[ENEM]+best_fit_std[ENEM], color='darkblue', alpha=0.5)
plt.plot(gens, means[ENEM], color='darkblue')
plt.fill_between(gens, means[ENEM]-means_std[ENEM], means[ENEM]+means_std[ENEM], color='darkblue', alpha=0.5)
plt.xlim(0, max(gens))
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=15)
plt.rcParams.update({'font.size':15})

custom_lines = [Line2D([0], [0], color='black'), Line2D([0], [0], color='black',linestyle='-.'),
               Line2D([0], [0], color='darkblue'), Line2D([0], [0], color='darkblue', linestyle = '-.')]
plt.legend(custom_lines, [r'$\mu, \lambda$ mean', r'$\mu, \lambda$ best', r'$\mu + \lambda$ mean', r'$\mu + \lambda$ best'],
           loc='lower right', framealpha=1, fontsize=fontsize)

# plt.subplots_adjust(hspace=0)
plt.xlabel('Generation', fontsize=fontsize)
plt.ylabel('Fitness', fontsize=fontsize)
plt.savefig(f'fitness_enemy{ENEM}.png', dpi=300)
plt.show()
