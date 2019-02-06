import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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

        # binary representations of ring and sites
        self._ring = 0
        self._ext = 0

        # seed the random generator
        if SEED > 0:
            np.random.seed(SEED)

        self.shuffle_state()

    def shuffle_state(self):
        """
        Shuffles the state of the necklace randomly.
        """
        # set the classes of the nodes
        a = []
        while len(a) < (self.__m*self.__n/2):
            x = np.random.randint(0,self.__m*self.__n)
            if x not in a:
                a.append(x)

        for x in a:
            if x < int(self.__n * self.__m / 2):
                self._ring ^= 1 << x
            if x >= int(self.__n * self.__m / 2):
                self._ext ^= 1 << (x - int(self.__n * self.__m / 2))

    def get_energy(self):
        """
        Faster method for getting the energy
        :return: Energy of the current necklace setup
        """

        # connection of ring with external
        ext_energy = bin(self._ring ^ self._ext).count('1')
        # connections in the ring
        ring_string = bin(self._ring)[2:].zfill(self.__m)
        shift_string = ring_string[-1] + ring_string[:-1]
        shifted_ring = int(shift_string,2)
        ring_energy = bin(self._ring ^ shifted_ring).count('1')

        energy = ext_energy + ring_energy
        return energy

    def pair_exchange_random(self):
        """
        Exchanges two randomly chosen nodes in the necklace.
        :return: pos1,pos2 / Positions of the two exchanged notes
        """

        # Choose first node for random exchange
        pos1 = np.random.randint(0,int(self.__m*self.__n))
        if pos1 >= int(self.__m*self.__n/2):
            val_pos1 = (1 << pos1-int(self.__m*self.__n/2)) & self._ext
            # Swap to other class
            self._ext = self._ext ^ (1 << pos1-int(self.__m*self.__n/2))
        else:
            val_pos1 = (1 << pos1) & self._ring
            # Swap to other class
            self._ring = self._ring ^ (1 << pos1)

        if val_pos1 > 0: val_pos1 = 1

        # choose other node until it has another class
        while True:
            pos2 = np.random.randint(0,int(self.__m*self.__n))
            if pos1 == pos2:
                continue

            #if it is a position in external
            if pos2 >= int(self.__m * self.__n / 2):
                #get value of that
                val_pos2 = (1 << pos2 - int(self.__m * self.__n / 2)) & self._ext
                if val_pos2 > 0: val_pos2 = 1
                if val_pos1 != val_pos2: #If two different classes
                    # Swap to other class
                    self._ext = self._ext ^ (1 << pos2 - int(self.__m * self.__n / 2))
                    break

            else:
                val_pos2 = (1 << pos2) & self._ring
                if val_pos2 > 0: val_pos2 = 1
                if val_pos1 != val_pos2:
                    self._ring = self._ring ^ (1 << pos2)
                    break

        #Store the last two positions of swapping
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

    def print(self):
        """
        Prints the current config of the necklace
        :return:
        """
        print('ring: ' + bin(self._ring)[2:].zfill(self.__m))
        print('ext:  ' + bin(self._ext)[2:].zfill(self.__m))

    def draw(self,filename,SEED=42):
        """
        Draws the necklace model. Not too beautiful right now but works.
        For color settings in networkx see.
        https://stackoverflow.com/questions/27030473/how-to-set-colors-for-nodes-in-networkx-python
        :return:
        """
        np.random.seed(SEED)
        G = nx.Graph()
        G.add_node(0) # Initial node
        #Ring connection
        for i in range(1,self.__m):
            G.add_node(i)
            G.add_edge(i-1,i)
        G.add_edge(0,self.__m-1) # Adds last ring connection

        # Connections to fully connected graphs
        for i in range(1,self.__n):
            for j in range(self.__m):
                G.add_node(j+i*self.__m)
                G.add_edge(j,j+i*self.__m)
        #colormap
        f = lambda x: 'red' if x == 0 else 'blue'
        cmap = [f(x) for x in self._sites]
        nx.draw(G,node_color=cmap)
        plt.text(0,0,'Energy: ' + str(self.get_energy()))
        plt.savefig(filename)
        plt.close()

