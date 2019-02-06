from necklace_model import Necklace
from simulated_annealing import Annealer
import numpy as np
import matplotlib.pyplot as plt

def necklace_test():
    n = 5
    MyNecklace = Necklace(n,2)
    MyNecklace.print()
    print(MyNecklace.get_energy())

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