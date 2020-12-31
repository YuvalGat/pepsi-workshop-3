import numpy as np
from physics import *


def T0_config(dt, N, L, rc):
    r = np.array([[np.random.uniform(high=L), np.random.uniform(high=L)] for i in range(N)])
    r_old = r
    stop_condition = True
    while stop_condition:
        r_new, v = verlet_step(r_old, r, dt, L, rc)
        r_old = r_new
        r = r_new
        r_old = np.remainder(r_old, [L, L])
        stop_condition = system_energy(r_old, r, r_new, dt, L, rc)[1] > 0.01
        r = np.remainder(r, [L, L])
    return r
