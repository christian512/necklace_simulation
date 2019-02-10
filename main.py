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
    MyNecklace = Necklace(6,2)
    MyAnnealer = Annealer()
    MyAnnealer.set_model(MyNecklace)
    MyAnnealer.run_adapted(max_steps=100)


if __name__ == '__main__':
    annealer_test()