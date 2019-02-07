import sys
import numpy as np
from necklace_model import Necklace
import random


class Annealer:
    """
    A class for running simulated annealing on a general model.
    """

    def __init__(self):
        self.__temps = []
        self.__model = Necklace(2,2)

    def set_temps(self, temps):
        self.__temps = temps

    def set_model(self, model):
        self.__model = model

    def run(self,ensemble_size=1):
        """
        Runs the simulated annealing on the model. With a given ensemble size.
        :return:
        """
        # Check if all functions/variables are set
        if len(self.__temps) == 0:
            sys.exit('Annealer: Temperatures not set, use set_temps')
        if self.__model == 0:
            sys.exit('Annealer: Model not set, use set_model(model)')

        #Create ensemble and choose random initial state for each
        ensemble = [self.__model]*ensemble_size
        for k in range(ensemble_size):
            ensemble[k].shuffle_state()

        # Run the simulated annealing method
        energyArr = np.empty(len(self.__temps))
        energyVBSFArr = np.empty(len(self.__temps))
        for i in range(len(self.__temps)):

            T = self.__temps[i]

            if i == 0:
                energyVBSFArr[i] = ensemble[0].get_energy()
            else:
                energyVBSFArr[i] = energyVBSFArr[i-1]

            e_sum = 0
            for k in range(ensemble_size):
                dE = 0 - ensemble[k].get_energy()
                e_sum += np.abs(dE)
                if np.abs(dE) < energyVBSFArr[i]:
                    energyVBSFArr[i] = np.abs(dE)
                ensemble[k].pair_exchange_random()
                dE += ensemble[k].get_energy()

                # Metropolis
                if dE < 0:
                    continue
                # calculate the boundary for accepting
                if T == np.inf:
                    p = 1
                elif T == 0:
                    p = 0
                else:
                    p = np.exp(-dE/T)
                r = random.random()
                # If change not accepted, go back to old model
                if r > p:
                    self.__model.undo_random_exchange()
            energyArr[i] = e_sum / ensemble_size
        return energyArr,energyVBSFArr


