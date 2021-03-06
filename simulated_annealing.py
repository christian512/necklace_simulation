import sys
import numpy as np
from scipy.sparse.linalg import eigsh
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

        # Create ensemble and choose random initial state for each
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
                    ensemble[k].undo_random_exchange()
            energyArr[i] = e_sum / ensemble_size
        return energyArr,energyVBSFArr

    def run_adapted(self,ensemble_size=1,therm_speed=1,start_temp=40,end_temp=0.5,max_steps=9999999,update_steps=1):
        """
        Running a simulated annealing process with adapted/optimal temperature schedule
        :param update_steps: Gives the number of steps until the temperature is updated
        :param therm_speed: Thermodynamic speed for the process
        :param max_steps: Maximum steps before it ends
        :param ensemble_size: Number of walkers for the process
        :param start_temp: Start temperature
        :param end_temp: End temperature
        :return: Mean energy, Best Energy , Temperature Schedule
        """
        if len(self.__temps) > 0:
            print('Temperatures set with Annealer.set_temps() are not used within Annealer.run_adapted()')

        # Create ensemble and choose random initial state for each
        ensemble = []
        for k in range(ensemble_size):
            nkl = self.__model.get_copy()
            nkl.shuffle_state()
            ensemble.append(nkl)

        # Initialize Q matrix
        if self.__model.dims_lumped > 1000:
            print('The used system is very large: Simulated annealer does not include sparse matrix implementation!')
        Q = np.zeros([self.__model.dims_lumped,self.__model.dims_lumped],dtype=int)

        # Set temperature, step counter and energy arrays
        T = start_temp
        step = 0
        energiesArr = np.empty(max_steps)
        energiesVBSF = np.empty(max_steps)
        temps = np.empty(max_steps)

        # Until end is reached
        while T >= end_temp and step < max_steps:
            # Set best energy from before
            if step == 0:
                energiesVBSF[step] = ensemble[0].get_energy()
            else:
                energiesVBSF[step] = energiesVBSF[step-1]

            # Store temperature
            temps[step] = T

            e_sum = 0 # For calculating the average energy
            # For each particle in the ensemble
            for l in range(ensemble_size):
                # Perform transition
                e_cur = ensemble[l].get_energy()

                # Energie calculations
                e_sum += e_cur
                if e_cur < energiesVBSF[step]:
                    energiesVBSF[step] = e_cur

                cur_state = ensemble[l].get_lumped_index(e_cur)
                ensemble[l].pair_exchange_random()
                e_new = ensemble[l].get_energy()
                new_state = ensemble[l].get_lumped_index(e_new)

                # Add entry in the transition matrix
                Q[new_state, cur_state] += 1

                # Metropolis algorithm
                de = e_new - e_cur
                if de < 0:
                    continue
                if T == np.inf:
                    p = 1
                elif T == 0:
                    p = 0
                else:
                    p = np.exp(-de/T)
                r = random.random()
                if r > p:
                    ensemble[l].undo_random_exchange()
            # Store energy
            energiesArr[step] = e_sum / ensemble_size

            # Check if temperature needs update
            if step % update_steps != 0:
                step += 1
                continue

            # Calculate the probability matrix
            P = Q / Q.sum(axis=0)
            P[np.isnan(P)] = 0



            # Square P until any row does not change anymore
            check = False
            runs = 0
            while check == False:
                runs += 1
                P_new = np.matmul(P, P)
                dists = np.abs(P-P_new)
                dists = np.sum(dists,axis=0)
                dists[dists==0] = np.max(dists)
                P = P_new
                if np.min(dists) < 10**-3:
                    idx = np.argmin(dists)
                    check = True
            # Get degeneracies from P
            degs = P[:,idx]
            # Get all energies
            energies = ensemble[0].allEnergies
            # Statistical quantities
            z = np.sum(degs*np.exp(-energies/T))
            if z == 0:
                step += 1
                continue

            e_mean = 1/z * np.sum(energies * degs * np.exp(-energies/T))
            heat_cap = 1/(T**2*z) * np.sum((energies-e_mean)**2*degs*np.exp(-energies/T))

            # Get the second largest eigenvalue for later relaxation time
            lambda2 = eigsh(P, 2, which='LA')[0][0]

            relax_time = -1/np.log(lambda2)
            # Update temperature
            T = T * (1-therm_speed/(relax_time*np.sqrt(heat_cap)))

            if T < 0:
                print('T = 0 reached. Programm ended')
                break

            step += 1

        if step < max_steps:
            temps = temps[:step]
            energiesArr = energiesArr[:step]
            energiesVBSF = energiesVBSF[:step]

        return energiesArr,energiesVBSF,temps,degs

