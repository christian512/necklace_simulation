import sys
import numpy as np
from necklace_model import Necklace


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

    def run(self,runs_per_temp=1):
        """
        Runs the simulated annealing on the model
        :return:
        """
        # Check if all functions/variables are set
        if len(self.__temps) == 0:
            sys.exit('Annealer: Temperatures not set, use set_temps')
        if self.__model == 0:
            sys.exit('Annealer: Model not set, use set_model(model)')

        # Run the simulated annealing method
        energyArr = np.empty(len(self.__temps))
        energyVBSFArr = np.empty(len(self.__temps))
        for i in range(len(self.__temps)):
            energyVBSFArr[i] = self.__model.get_energy()
            Etemp = 0
            for k in range(runs_per_temp):
                T = self.__temps[i]
                dE = 0 - self.__model.get_energy()
                Etemp += np.abs(dE)
                if np.abs(dE) < energyVBSFArr[i]:
                    energyVBSFArr[i] = np.abs(dE)
                self.__model.pair_exchange_random()
                dE += self.__model.get_energy()
                # If energy got lower, go to next
                if dE < 0:
                    continue

                # calculate the boundary for accepting
                if T == np.inf:
                    p = 1
                elif T == 0:
                    p = 0
                else:
                    p = np.exp(-dE/T)
                r = np.random.rand()
                # If change not accepted, go back to old model
                if r > p:
                    self.__model.undo_random_exchange()
            energyArr[i] = Etemp / runs_per_temp
        return energyArr,energyVBSFArr


