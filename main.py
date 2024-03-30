import json
import time

import pandas as pd
from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.creators.ga_creators.int_vector_creator import GAIntVectorCreator
from eckity.breeders.simple_breeder import SimpleBreeder

from eckity.genetic_operators.crossovers.vector_k_point_crossover import VectorKPointsCrossover
from eckity.genetic_operators.mutations.vector_random_mutation import IntVectorNPointMutation
from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
from eckity.statistics.best_average_worst_statistics import BestAverageWorstStatistics
from eckity.subpopulation import Subpopulation
from FitnessEvaluator import FitnessEvaluator
from Analytics import Analytics
from EvaluatorMetaData import *

def executor(num_of_generations, population_size, crossover, mutation, x_values_bounds):
    output_file = open("output.txt", "w")
    #functions_generator.generate_functions(x_values_bounds)
    # Initialize the evolutionary algorithm
    algo = SimpleEvolution(
        Subpopulation(creators=GAIntVectorCreator(length=2, bounds=x_values_bounds),
                      population_size=population_size,
                      evaluator=FitnessEvaluator(F1,F2,G1,G2,C),
                      higher_is_better=False,  # False because we look for minimum.
                      elitism_rate=0.0,
                      operators_sequence=[
                          VectorKPointsCrossover(probability=crossover, k=1),
                          IntVectorNPointMutation(probability=mutation, n=1)  # changes only one of x1,x2.(n parameter)
                      ],
                      selection_methods=[
                          # (selection method, selection probability) tuple
                          (TournamentSelection(
                              tournament_size=4, higher_is_better=False), 1)
                      ]),
        breeder=SimpleBreeder(),
        max_workers=1,
        max_generation=num_of_generations,
        statistics=BestAverageWorstStatistics(output_stream=output_file)
    )

    algo.evolve()
    best_numbers = algo.execute()
    output_file.close()
    return best_numbers


def calculate_average_differences(results):
    total_diff = 0
    for result in results:
        total_diff += result['difference']

    return total_diff / len(results)


def append_dict_to_file(dictionary, filename):
    with open(filename, 'a') as file:
        file.write(json.dumps(dictionary) + '\n')


if __name__ == '__main__':

    # # num_of_generations , population_size, crossover_rate, mutation_rate
    generation_list = []
    population_list = []
    mutation_list = []
    crossover_list = []
    convergence_list = []
    best_fitness_list = []
    #Those lists are for the excel sheet.
    results_list = []
    for i in range(NUM_OF_ITERATIONS):
        chosen_set = PARAMETERS_SETS[i]
        if NUM_OF_ITERATIONS == 1:
            chosen_set = CHOSEN_SET
        num_of_generations = chosen_set[0]
        population_size = chosen_set[1]
        crossover = chosen_set[2]
        mutation = chosen_set[3]
        random.seed()
        print("Starting new iteration")
        actions = executor(num_of_generations, population_size, crossover, mutation, X_BOUNDS)
        analytics = Analytics(X_BOUNDS)
        analytics.read_file()
        #print(f'x1: {actions[0]}, x2: {actions[1]}, value: {analytics.best_fitness}')
        if STATISTICS:
            # plot graph and print stats
            generation_list.append(num_of_generations)
            population_list.append(population_size)
            mutation_list.append(mutation)
            crossover_list.append(crossover)
            best_fitness_list.append(analytics.best_fitness)
            convergence_list.append(analytics.converge_gen_number)
            #analytics.plot_graph(i, PARAMETERS_SETS[i], analytics.best_fitness)
            current_result = {
                'actions': actions,
                'best_fitness': analytics.best_fitness,
                'real_min_points_values': analytics.real_min_points_values,
                'real_min_val': analytics.real_min_val,
                'difference': abs(analytics.real_min_val - analytics.best_fitness)
            }
            print(current_result)
    if STATISTICS:
        excel_dict = {
            'num_of_generations': generation_list,
            'population_size': population_list,
            'crossover_rate': crossover_list,
            'mutation_rate': mutation_list,
            'best_fitness': best_fitness_list,
            'converge_gen_number': convergence_list,
        }
        new_data = pd.DataFrame(excel_dict)
        # Specify the file path for the new Excel file
        new_excel_file = f'RUN{time.time()}.xlsx'
        with pd.ExcelWriter(new_excel_file) as writer:
            # Write the new data to a new sheet named 'NewSheet'
            new_data.to_excel(writer, sheet_name='NewSheet',index = False)
