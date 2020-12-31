import numpy as np
from physics import *
from matplotlib import pyplot as plt


def velocity(old, new, dt):
    return np.sum(np.abs((new - old)) / (2 * dt))


def T0_config(dt, N, L, rc):
    r = np.random.rand(N, 2) * L
    r_old = r
    stop_condition = True
    temperature = []
    pressure = []
    energy = []
    counter = []
    i = 1
    while stop_condition:
        ver, vir = verlet_step(r_old, r, dt, L, rc)
        r_new = np.remainder(ver, L)
        stop_condition = velocity(r_old, r_new, dt) > 0.01
        r_old = r_new
        if i % 1000 == 0:
            counter.append(i)
            ek, up, etot = system_energy(r_old, r, r_new, dt, L, rc)
            p = pressure_virial(vir, ek, L)
            pressure.append(p)
            energy.append(ek)
            temperature.append(p * L * L * L / (N * k))
        r = r_new
        r_old = np.remainder(r_old, [L, L])
        i += 1
    plt.plot(counter, temperature)
    # plt.plot(counter, pressure)
    return r
