from necklace_model import Necklace
from simulated_annealing import Annealer
import numpy as np
import matplotlib.pyplot as plt

def necklace_test():
    n = 23
    MyNecklace = Necklace(n,2)
    for i in range(10000):
        if MyNecklace.get_energy() not in MyNecklace._allEnergies:
            print('Energy not in all energies: '+ str(MyNecklace.get_energy()))
            MyNecklace.print()
            break
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