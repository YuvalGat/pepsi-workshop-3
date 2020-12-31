import numpy as np
from numba import jit

m = 10e-27
k = 1.38 * 10e-23


# r_vec is a np.array of D elements (D is the number of dimensions).
# it is the vector which points from particle 1 to particle 2 (=r_ij=ri-rj)
# #this function returns the Lennard-Jones potential between the two particles
@jit
def LennardJonesPotential(r_vec, rc):
    r = np.linalg.norm(r_vec)  # calculate norm (= | r_ij |)
    if r > rc: return 0.
    rc_3 = rc * rc * rc
    rc_6 = rc_3 * rc_3
    rc_12 = rc_6 * rc_6
    r_3 = r * r * r
    r_6 = r_3 * r_3
    r_12 = r_6 * r_6
    return 4 * (1 / r_12 - 1 / r_6) - 4 * (1 / rc_12 - 1 / rc_6)


# same as previous method but returns the force between the two particles #this is the gradient of the previous method
@jit
def LennardJonesForce_fast(r_vec, rc):
    r = np.linalg.norm(r_vec)  # calculate norm (= | r_ij |)
    if r > rc:
        return np.zeros_like(r_vec)
    r_pow2 = r * r
    r_pow4 = r_pow2 * r_pow2
    r_pow8 = r_pow4 * r_pow4
    r_pow6 = r_pow2 * r_pow4
    return 24 / r_pow8 * (6 / r_pow6 - 2) * r_vec


# this method calculates the total force on each particle
# r is a 2D array where r[i,:] is a vector with length D (dimensions)
# which represents the position of the i-th particle
# (in 2D case r[i,0] is the x coordinate and r[i,1] is the y coordinate of the i-th
# this function returns a numpy array F of the same dimesnsions as r
# where F[i,:] is a vector which represents the force that acts on the i-th particle
# this function also returns the virial
@jit
def LJ_Forces(r, L=2, rc=3):
    F = np.zeros_like(r)
    virial = 0
    N = r.shape[0]  # number of particles
    # loop on all pairs of particles i , j
    for i in range(1, N):
        for j in range(i):
            r_ij = r[i, :] - r[j, :]
            r_ij = r_ij - L * np.rint(r_ij / L)  # see class on boundary conditions
            rd = np.linalg.norm(r_ij)  # calculate norm (= | r_ij |)
            if rd > rc:
                f_ij = np.zeros(len(r_ij))
            else:
                r_2 = rd * rd
                r_4 = r_2 * r_2
                r_8 = r_4 * r_4
                r_14 = r_2 * r_4 * r_8
                f_ij = (4 * (12 / r_14 - 6 / r_8)) * r_ij
            F[i, :] += f_ij
            F[j, :] -= f_ij  # third law of newton
            virial += np.dot(f_ij, r_ij)
    return F, virial


@jit
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


# @jit
def verlet_step(r_old, r, dt, L, rc):
    F, virial = LJ_Forces(r, L, rc)
    a = F / m
    r_new = 2 * r + a * dt * dt - r_old

    return r_new, virial


@jit
def pressure_virial(virial, Ek, L):
    V_inv_third = 1 / (3 * L * L * L)
    P = V_inv_third * (Ek + virial)
    return P
