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

        self._sites = np.zeros(n*m,dtype=int)  # Contains ring sites the first m elements and then the connected graphs
        self._sites[0:int((m*n)/2)] = np.array([1]*int((m*n)/2))  # set the first elements to class one

        # shuffle the array to have random initialization
        if SEED > 0:
            np.random.seed(SEED)
        self._sites = np.random.permutation(self._sites)

    def shuffle_state(self):
        """
        Shuffles the state of the necklace randomly.
        """
        self._sites = np.random.permutation(self._sites)

    def get_energy(self):
        """
        Faster method for getting the energy
        :return: Energy of the current necklace setup
        """
        # Ring connections
        ring = self._sites[0:self.__m]
        #shifted_ring = np.roll(ring,1) # Roll is somehow really slow
        shifted_ring = np.concatenate([ring[-1:],ring[:-1]])
        energy = ring.shape[0] - np.equal(ring,shifted_ring).sum()
        # External connections
        ext = self._sites[self.__m:self._sites.shape[0]]
        energy += ring.shape[0] - np.equal(ring,ext).sum()
        return energy

    def pair_exchange_random(self):
        """
        Exchanges two randomly chosen nodes in the necklace.
        :return: pos1,pos2 / Positions of the two exchanged notes
        """
        pos1 = np.random.randint(0,int(self.__m*self.__n))
        pos2 = np.random.randint(0,int(self.__m*self.__n))


        #Both nodes should have different classes
        while self._sites[pos1] == self._sites[pos2]:
            pos2 = np.random.randint(0,int(self.__m*self.__n))
        # Swap the classes
        temp = self._sites[pos1]
        self._sites[pos1] = self._sites[pos2]
        self._sites[pos2] = temp

        self.__lastPos1 = pos1
        self.__lastPos2 = pos2

        return pos1,pos2

    def undo_random_exchange(self):
        """
        Undoes the previous random exchange
        :return:
        """
        temp = self._sites[self.__lastPos1]
        self._sites[self.__lastPos1] = self._sites[self.__lastPos2]
        self._sites[self.__lastPos2] = temp

    def pair_exchange(self,pos1,pos2):
        """
        Exchanges two given positions in the necklace
        :param pos1: position of first node
        :param pos2: position of second noce
        """
        temp = self._sites[pos1]
        self._sites[pos1] = self._sites[pos2]
        self._sites[pos2] = temp



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
