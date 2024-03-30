import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from EvaluatorMetaData import F1,F2,G1,G2,C

class Analytics:
    """class for graph ploting"""

    def __init__(self,bounds):
        self.all_best_fitness = []
        self.all_worst_fitness = []
        self.all_average_fitness = []
        self.num_of_gen = []
        self.best_fitness = sys.maxsize
        self.converge_gen_number = 0
        self.real_min_points_values = (0,0)
        self.real_min_val = 0
        self.find_real_min_values(bounds)

    # TODO: Change best fitness
    def read_file(self):
        with open("output.txt", "r") as my_file:
            lines = my_file.readlines()
            j = 0
            for i in range(2, len(lines), 6):
                best_fitness = float(lines[i][13:].strip())
                self.all_best_fitness.append(best_fitness)
                if int(best_fitness) < int(self.best_fitness):
                    self.converge_gen_number = j
                    self.best_fitness = best_fitness
                worst_fitness = float(lines[i + 1][14:].strip())
                self.all_worst_fitness.append(worst_fitness)
                average_fitness_str = lines[i + 2][16:].strip()
                if '/' in average_fitness_str:
                    num_str, divider_str = average_fitness_str.split('/')
                    # Convert numerator and denominator to floats
                    num = float(num_str)
                    divider = float(divider_str)
                    average_fitness = num / divider
                else:
                    average_fitness = float(average_fitness_str)

                self.all_average_fitness.append(average_fitness)
                self.num_of_gen.append(j)
                j += 1

    def plot_graph(self, iteration, parameters_set, best_fitness):
        population_size = parameters_set[1]
        crossover = parameters_set[2]
        mutation = parameters_set[3]
        # x axis values
        x = self.num_of_gen
        # corresponding y axis values
        y1 = self.all_best_fitness
        plt.figure()
        # plotting the points
        plt.plot(x, y1, color='blue', label="Best Fitness")

        plt.xlabel('Number of generations')
        plt.ylabel('Fitness values')
        plt.title(
            f'pop: {population_size}; mut: {mutation}; CO: {crossover}; BFit: {int(best_fitness)}; Conv: {self.converge_gen_number}')
        plt.tight_layout()
        # show a legend on the plot
        plt.legend()
        folder = f'population_size_{population_size}/'
        os.makedirs(folder, exist_ok=True)
        plt.savefig(os.path.join(folder, f'output{iteration}.png'))
        plt.close()
        # function to show the plot
        # plt.show()

    def find_real_min_values(self, limit):
        # Define the range of x1 and x2
        x_range = np.arange(limit[0], limit[1])
        # Initialize variables to store the minimum value and corresponding x1, x2
        min_value = float('inf')
        min_x1, min_x2 = None, None
        # Brute-force search over the grid of points
        for x1 in x_range:
            for x2 in x_range:
                if self.compute_g(x1,x2):
                    value = self.compute_f(x1,x2)
                    if value < min_value:
                        min_value = value
                        min_x1 = x1
                        min_x2 = x2
        # Output the result
        self.real_min_points_values = (min_x1, min_x2)
        self.real_min_val = min_value

    def compute_f(self, x1, x2):
        if isinstance(F1, dict):
            return F1[x1] + F2[x2]

        return F1(x1) + F2(x2)

        # BOOL

    def compute_g(self, x1, x2):
        if isinstance(G1, dict):
            output = G1[x1] + G2[x2]
        else:
            output = G1(x1) + G2(x2)
        return output <= C