import numpy as np
from numba import njit

m = 10e-27
k = 1.38 * 10e-23


# r_vec is a np.array of D elements (D is the number of dimensions).
# it is the vector which points from particle 1 to particle 2 (=r_ij=ri-rj)
# #this function returns the Lennard-Jones potential between the two particles
def LennardJonesPotential(r_vec, rc):
    r = np.linalg.norm(r_vec)  # calculate norm (= |r_ij|)
    if r > rc: return 0.
    VLJ_rc = 4 * (1 / rc ** 12 - 1 / rc ** 6)
    return 4 * (1 / r ** 12 - 1 / r ** 6) - VLJ_rc


# same as previous method but returns the force between the two particles #this is the gradient of the previous method
def LennardJonesForce(r_vec, rc):
    r = np.linalg.norm(r_vec)  # calculate norm (= |r_ij|)
    if r > rc: return 0. * r_vec
    return 4 * (12 / r ** 14 - 6 / r ** 8) * r_vec  # calculate the gradient of "LennardJ


# this method calculates the total force on each particle
# r is a 2D array where r[i,:] is a vector with length D (dimensions)
# which represents the position of the i-th particle
# (in 2D case r[i,0] is the x coordinate and r[i,1] is the y coordinate of the i-th
# this function returns a numpy array F of the same dimesnsions as r
# where F[i,:] is a vector which represents the force that acts on the i-th particle
# this function also returns the virial
def LJ_Forces(r, L, rc):
    F = np.zeros_like(r)
    virial = 0
    N = r.shape[0]  # number of particles
    # loop on all pairs of particles i, j
    for i in range(1, N):
        for j in range(i):
            r_ij = r[i, :] - r[j, :]
            r_ij = r_ij - L * np.rint(r_ij / L)  # see class on boundary
            f_ij = LennardJonesForce(r_ij, rc)
            F[i, :] += f_ij
            F[j, :] -= f_ij
            virial += np.dot(f_ij, r_ij)  # see class on virial theorem
    return F, virial


@njit
def system_energy(r_old, r, r_new, dt, L, rc):
    v = (r_new - r_old) / (2 * dt)  # Second order linear approximation of derivative
    e_ks = v * v  # Kinetic energy for each particle (/1.5m)
    e_k = 1.5 * m * np.sum(e_ks)  # Total kinetic energy

    # u_p = np.sum(LennardJonesPotential(r, rc))
    N = r.shape[0]  # number of particles
    u_p = 0
    # loop on all pairs of particles i, j
    for i in range(1, N):
        for j in range(i):
            r_ij = r[i, :] - r[j, :]
            r_ij = r_ij - L * np.rint(r_ij / L)
            u_ij = LennardJonesPotential(r_ij, rc)
            u_p += u_ij

    e_tot = e_k + u_p

    return e_k, u_p, e_tot


@njit
def verlet_step(r_old, r, dt, L, rc):
    F, virial = LJ_Forces(r, L, rc)
    a = F / m
    r_new = 2 * r + a * dt * dt - r_old

    return r_new, virial


@njit
def pressure_virial(virial, Ek, L):
    V_inv_third = 1 / (3 * L * L * L)
    P = V_inv_third * (Ek + virial)
    return P
