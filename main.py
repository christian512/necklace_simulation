from necklace_model import Necklace
from simulated_annealing import Annealer
import numpy as np
import matplotlib.pyplot as plt

def necklace_test():
    n = 3
    MyNecklace = Necklace(n,2)
    MyNecklace.print()
    for i in range(10000):
        MyNecklace.pair_exchange_random()
    MyNecklace.print()

def annealer_test():
    MyNecklace = Necklace(20,2)
    MyAnnealer = Annealer()
    MyAnnealer.set_model(MyNecklace)
    energies,energiesVBSF,temps = MyAnnealer.run_adapted(max_steps=10000,ensemble_size=100,therm_speed=10**-4)
    plt.plot(temps)
    print(temps)
    plt.savefig('test.png')

if __name__ == '__main__':
    annealer_test()