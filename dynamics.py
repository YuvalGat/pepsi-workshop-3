import numpy as np
from physics import *
from matplotlib import pyplot as plt

energy = []
pressure = []
temperature = []
counter = []


def T0_config(dt, N, L, rc):
    r = np.array([[np.random.uniform(high=L), np.random.uniform(high=L)] for i in range(N)])
    plt.scatter(r[:, 0], r[:, 1])
    plt.show()
    r_old = r
    # stop_condition = True
    i = 0
    while i < 100:
        r_new, v = verlet_step(r_old, r, dt, L, rc)
        r_old = r_new
        ek, up, etot = system_energy(r_old, r, r_new, dt, L, rc)
        p = pressure_virial(v, ek, L)

        stop_condition = etot > 0.0001
        r = r_new
        # r_old = np.remainder(r_old, [L, L])
        # r = np.remainder(r, [L, L])
        i += 1
        if i % 1000 == 0:
            energy.append(ek)
            pressure.append(p)
        print(i)
    return r
