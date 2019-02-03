from necklace_model import Necklace
from simulated_annealing import Annealer
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def necklace_test():
    MyNecklace = Necklace(20,2)
    N = 1
    n = 100000
    for i in range(100000):
        MyNecklace.pair_exchange_random()


def annealer_test():
    MyNecklace = Necklace(20,2)
    MyAnnealer = Annealer()
    temps = np.array([np.inf]*10000)
    MyAnnealer.set_temps(temps)
    MyAnnealer.set_model(MyNecklace)
    energyArr = MyAnnealer.run()
    plt.plot(energyArr)
    plt.savefig('test.png')


if __name__ == '__main__':
    necklace_test()