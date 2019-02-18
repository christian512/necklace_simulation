from necklace_model import Necklace
import random
import numpy as np
import matplotlib.pyplot as plt

class GeneticAlgorithm():
    def __init__(self):

        # Set standard model
        self.__model = Necklace(4, 1)
        population = [self.__model]

        # Rates for updating population

    def set_model(self, model):
        self.__model = model

    def run(self,population_size=100,num_gens=1,crossover_rate=0.1,mutation_rate=0.1,clone_rate=0.1):
        """
        Runs the genetic algorithm with given parameters
        :return:
        """
        # Create population
        population = []
        for k in range(population_size):
            nkl = self.__model.get_copy()
            nkl.shuffle_state()
            population.append(nkl)

        energiesArr = np.empty(num_gens)
        energiesVBSFArr = np.empty(num_gens)

        # Iterate over all generations
        for o in range(num_gens):

            # Crossovers
            idx_cross1 = random.sample(range(0,population_size),int(crossover_rate/2*population_size))
            idx_cross2 = random.sample(range(0,population_size),int(crossover_rate/2*population_size))
            for i,j in zip(idx_cross1,idx_cross2):
                population.append(population[i].get_copy())
                population.append(population[j].get_copy())
                population[i].crossover(population[j])

            # Mutants
            idx_mutants = random.sample(range(0,population_size),int(mutation_rate*population_size))
            for i in idx_mutants:
                population.append(population[i].get_copy())
                population[i].mutate()

            # Clone the individuals with lowest energy
            pop_energies = [x.get_energy() for x in population]
            sort_idx = np.argsort(pop_energies)
            for i in range(int(clone_rate*population_size)):
                population.append(population[sort_idx[i]].get_copy())
                pop_energies.append(population[sort_idx[i]].get_energy())

            # Reduce population size to original one
            pop_energies = [indiv.get_energy() for indiv in population]
            sort_idx = np.argsort(pop_energies)
            new_generation = []
            sum_energy = 0
            for i in range(population_size):
                new_generation.append(population[sort_idx[i]])
                sum_energy += pop_energies[sort_idx[i]]
            population = new_generation

            # Set energies
            energiesArr[o] = sum_energy / population_size
            if o == 0: energiesVBSFArr[o] = np.min(pop_energies)
            elif energiesVBSFArr[o-1] > np.min(pop_energies): energiesVBSFArr[o] = np.min(pop_energies)
            else: energiesVBSFArr[o] = energiesVBSFArr[o-1]

        return energiesArr, energiesVBSFArr


if __name__ == '__main__':
    nkl = Necklace(20,2)
    ga = GeneticAlgorithm()
    ga.set_model(nkl)
    energies,energiesVBSF = ga.run(population_size=1000,num_gens=100,clone_rate=0.5,crossover_rate=0.5,mutation_rate=0.5)
    plt.plot(energies)
    plt.savefig('test.png')
    plt.close()
    plt.plot(energiesVBSF)
    plt.savefig('test2.png')
