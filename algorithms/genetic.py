""" 
Genetic implementation of travelling salesman problem (TSP)

Steps
Init population 
Determine fitness
Select mating pool
Breed 
Mutate
Repeat
"""
import random
import json
import numpy as np
from matplotlib import pyplot as plt

from collections import namedtuple
from itertools import permutations, accumulate
from operator import itemgetter

from shared import EuclidianDist

City = namedtuple('City', ['x', 'y'])

def calculate_distance_from_next_city(route, from_i, to_i):
    from_city = route[from_i]
    to_city = route[to_i]
    return EuclidianDist(from_city, to_city)

def calculate_total_distance(route):
    route_len = len(route)
    return sum([
        calculate_distance_from_next_city(route, i, i + 1 if i + 1 < route_len else 0)
        for i in range(route_len)
    ], 0.0)

def calculate_fitness(route):
    """ For the travelling salesman problem fitness is a ratio from the distance
        between cities in a route
    """
    distance = calculate_total_distance(route)
    return 1 / distance


def gen_cities(num_cities, max_value = 500):
    return [
        City(int(random.random() * max_value), int(random.random() * max_value))
        for _ in range(num_cities)
    ]

def init_population(population_size, cities):
    """ Generate an array of length population_size of randomized values in cities """
    return [
        random.sample(cities, len(cities))
        for _ in range(population_size)
    ]


def rank_fitness(population):
    """ Sort routes in population based on fitness """
    results = {}
    for i, route in enumerate(population):
        results[i] = calculate_fitness(route)
    return sorted(results.items(), key=itemgetter(1), reverse=True)

def selection(ranked_population, num_seeded = 2):
    """
    Get indices to use in mating pool
    Uses the roulette wheel selection model
    """
    cumulative_sum = list(accumulate(map(itemgetter(1), ranked_population)))
    # Sum of all fitness
    sum_ = cumulative_sum[-1]
    cumulative_pct = [100 * sub_sum / sum_ for sub_sum in cumulative_sum]
    # First get indices of seeded values
    results = [
        ranked_population[i][0]
        for i in range(num_seeded)
    ]
    # For the rest of the population size gen a probality of picking a value
    # and get the first matching fitness less than or equal to the probability
    for _ in range(0, len(ranked_population) - num_seeded):
        probability = 100 * random.random()
        for i in range(len(ranked_population)):
            if probability <= cumulative_pct[i]:
                results.append(ranked_population[i][0])
                break
    
    return results

def gen_mating_pool(population, selected_results):
    """ Get actual population values for selected indices """
    mating_pool = [
        population[index]
        for index in selected_results
    ]
    return mating_pool

def breed_parents(parent1, parent2):
    """ 
    Get maximum and minimum fitness values in parent 1 
    Take all values that are range the range of min and max fitness 
    For parent 2 take only its children that dont exist in parent 1
    """
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent2))

    first_gene = min(geneA, geneB)
    last_gene = min(geneA, geneB)

    parent1_children = [
        parent1[i]
        for i in range(first_gene, last_gene)
    ]
    parent2_children = [val for val in parent2 if val not in parent1_children ]

    return parent1_children + parent2_children

def breed_population(mating_pool, num_seeds):
    """ Create breeding pool """
    # Randomize mating pool order
    pool = random.sample(mating_pool, len(mating_pool))
    pool_length = len(mating_pool)
    children = [
        mating_pool[i]
        for i in range(min([num_seeds, len(mating_pool)]))
    ]
    children.extend([
        breed_parents(pool[i], pool[pool_length - i - 1])
        for i in range(pool_length - num_seeds)
    ])

    return children

def mutate_route(route, mutation_rate):
    """ Mutation by swapping cities in a route """
    for index_swapped in range(len(route)):
        if random.random() < mutation_rate:
            index_swapped_with = random.randrange(0, len(route))
            route[index_swapped], route[index_swapped_with] = route[index_swapped_with], route[index_swapped]
    
    return route

def mutate_population(population, mutation_rate):
    return [
        mutate_route(route, mutation_rate)
        for route in population
    ]

def gen_next_generation(population, num_seeds, mutation_rate):
    ranked_population = rank_fitness(population)
    selected_indexes = selection(ranked_population)
    mating_pool = gen_mating_pool(population, selected_indexes)
    children = breed_population(mating_pool, num_seeds)
    return mutate_population(children, mutation_rate)


def genetic_algorithm(orig_population, population_size, num_seeds, mutation_rate, num_generations):
    population = init_population(population_size, orig_population)

    for _ in range(num_generations):
        population = gen_next_generation(population, num_seeds, mutation_rate)

    best_route_index = rank_fitness(population)[0][0]
    return population[best_route_index]

def genetic_algorithm_plot(orig_population, population_size, num_seeds, mutation_rate, num_generations):
    population = init_population(population_size, orig_population)
    best_routes_over_time = [ 1 / rank_fitness(population)[0][1] ]
    for _ in range(num_generations):
        population = gen_next_generation(population, num_seeds, mutation_rate)
        best_routes_over_time.append( 1 / rank_fitness(population)[0][1] )
    plt.plot(best_routes_over_time)
    plt.xlabel('Distance')
    plt.ylabel('Generation')
    plt.show()

def main():
    # import pdb; pdb.set_trace()
    cities = gen_cities(25, 200)
    genetic_algorithm_plot(cities, 
        population_size=100, 
        num_seeds=20, 
        mutation_rate=0.01, 
        num_generations=500)



if __name__ == "__main__":
    main()