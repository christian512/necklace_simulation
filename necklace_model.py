import sys
import numpy as np
import random
from scipy.special import binom

class Necklace:
    """
    A class that simulates the behavior of necklace models.
    More to come later
    """

    def __init__(self, m, n=2,SEED=0):
        """
        Initializes the necklace model
        :param m: Number of sites in the ring
        :param n: Number of nodes in each fully connected graph on the ring
        """
        # currently only n = 2,1 is implemented
        if n > 2:
            sys.exit("Initialization of Necklace: Currently only n < 3 is implemented")
        if n * m % 2 != 0:
            sys.exit('Initialization of Necklace: The number of sites is not even. Please give even number of sites!')

        # Store size variables
        self.__m = m
        self.__n = n

        # Initialize last swapped nodes
        self.__lastPos1 = 0
        self.__lastPos2 = 0

        # All energies that the necklace can take
        if self.__m % 2 == 0:
            self.allEnergies = np.arange(2, self.__m * 2 + 1, 2)
        else:
            self.allEnergies = np.arange(2, self.__m * 2 + 1, 1)

        # Set dimension of state space and lumped state space
        self.dims_states = binom(int(self.__m*self.__n),int(self.__m*self.__n/2))
        self.dims_lumped = len(self.allEnergies)

        # Binary representations of ring and sites
        self._ring = 0
        self._ext = 0

        # Seed the random generator
        if SEED > 0:
            random.seed = SEED

        self.shuffle_state()

    def shuffle_state(self):
        """
        Shuffles the state of the necklace randomly.
        """
        # set the classes of the nodes
        a = []
        while len(a) < (self.__m*self.__n/2):
            x = int(self.__m*self.__n*random.random())
            if x not in a:
                a.append(x)

        self._ring = 0
        self._ext = 0

        for x in a:
            self.change_class(x)

    def get_energy(self):
        """
        Method for getting the energy of the current necklace setup
        :return: Energy of the current necklace setup
        """

        # Connection of ring with external
        ext_energy = bin(self._ring ^ self._ext).count('1')

        # Connections in the ring
        ring_string = bin(self._ring)[2:].zfill(self.__m)
        shift_string = ring_string[-1] + ring_string[:-1]
        shifted_ring = int(shift_string,2)
        ring_energy = bin(self._ring ^ shifted_ring).count('1')

        # Give a quadratic energy penalty if the number of nodes of both classes is not equal
        c1 = bin(self._ring).count('1') + bin(self._ext).count('1')
        c1_exp = int(self.__m*self.__n/2)

        # Sum energy and return
        energy = ext_energy + ring_energy + (c1-c1_exp)**2
        return energy

    def val_at_pos(self,pos):
        """
        Returns the value/class (0 or 1) at a given position in the necklace
        :param pos: Position
        :return: Class of node at position pos
        """
        if pos >= self.__m*self.__n:
            sys.exit('Position in val_at_pos too big!')

        # If position in external ring
        if pos >= int(self.__m*self.__n/2):
            pos = pos - int(self.__m*self.__n/2)
            val = (1 << pos) & self._ext
        else:
            val = (1 << pos) & self._ring
        if val > 0: val = 1
        return val

    def change_class(self,pos):
        """
        Changes the class of node at position pos
        :param pos: position
        """
        if pos >= self.__m*self.__n:
            sys.exit('Pos too big in change_class!')

        if pos >= self.__m*self.__n/2:
            pos = pos - int(self.__m*self.__n/2)
            self._ext = (1 << pos) ^ self._ext
        else:
            self._ring = (1 << pos) ^ self._ring

    def pair_exchange_random(self):
        """
        Exchanges the classes of two random nodes in the necklace
        :return:
        """
        pos1 = int(self.__m*self.__n*random.random())
        pos2 = int(self.__m*self.__n*random.random())
        val1 = self.val_at_pos(pos1)
        val2 = self.val_at_pos(pos2)
        while pos1 == pos2 or val1 == val2:
            pos2 = int(self.__m*self.__n*random.random())
            val2 = self.val_at_pos(pos2)

        self.pair_exchange(pos1,pos2)
        self.__lastPos1 = pos1
        self.__lastPos2 = pos2

    def undo_random_exchange(self):
        """
        Undoes the previous random exchange
        :return:
        """
        self.pair_exchange(self.__lastPos1,self.__lastPos2)

    def pair_exchange(self,pos1,pos2):
        """
        Exchanges two given positions in the necklace
        :param pos1: position of first node
        :param pos2: position of second noce
        """
        if pos1 >= int(self.__m * self.__n / 2):
            self._ext = self._ext ^ (1 << pos1 - int(self.__m * self.__n / 2))
        else:
            self._ring = self._ring ^ (1 << pos1)

        if pos2 >= int(self.__m * self.__n / 2):
            self._ext = self._ext ^ (1 << pos2 - int(self.__m * self.__n / 2))
        else:
            self._ring = self._ring ^ (1 << pos2)

    def get_lumped_index(self,e=0):
        """
        Returns the index of the state in the lumped model
        :param e: Energy of the state, if not set it will be calculated automatically
        :return:
        """
        if e == 0:
            e = self.get_energy()
        res = np.where(self.allEnergies == e)
        if len(res[0]) == 0:
            print('Could not find energy ' + str(e))
            print('In array: ' + str(self.allEnergies))
            self.print()
            self.undo_random_exchange()
            self.print()
            sys.exit()
        return res[0][0]

    def get_free_energy(self,degeneracies):
        """
        Gets the current free energy of the necklace
        :return: Free energy
        """
        energy = self.get_energy()
        idx = self.get_lumped_index(e=energy)
        deg = degeneracies[idx]
        # TODO: deg is currently normalized, should multiply total number of states for real degs?

    def print(self,inline=False):
        """
        Prints the current config of the necklace
        :return:
        """
        if inline:
            out = bin(self._ring)[2:].zfill(self.__m)[::-1]
            out += bin(self._ext)[2:].zfill(self.__m)[::-1]
            print('nkl: ' + out)
        else:
            print('ring: ' + bin(self._ring)[2:].zfill(self.__m)[::-1])
            print('ext:  ' + bin(self._ext)[2:].zfill(self.__m)[::-1])

    def crossover(self,nkl):
        """
        Performs a crossing (genetic algorithm) of two necklaces
        :param nkl: Other necklace
        :return: New second necklace
        """
        randInt = int((self.__m*self.__n-2)*random.random())+1
        swap_pos = np.arange(randInt,self.__m*self.__n,dtype=int)
        for x in swap_pos:
            if self.val_at_pos(x) != nkl.val_at_pos(x):
                self.change_class(x)
                nkl.change_class(x)
        return nkl

    def mutate(self):
        """
        Mutates a random bit of the necklaces to the other class
        """
        randInt = int(self.__m*self.__n*random.random())
        self.change_class(randInt)

    def get_copy(self):
        """Returns a copy of it self."""
        nkl = Necklace(self.__m,self.__n)
        nkl._ring = self._ring
        nkl._ext = self._ext
        return nkl


if __name__ == '__main__':
    nkl = Necklace(4,2)
    nkl2 = Necklace(4,2)
    nkl.print(inline=True)
    nkl2.print(inline=True)
    print('+++ Crossover +++')
    nkl.crossover(nkl2)
    nkl.print(inline=True)
    nkl2.print(inline=True)
