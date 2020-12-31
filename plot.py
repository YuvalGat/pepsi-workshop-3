import numpy as np
import matplotlib.pyplot as plt
from physics import lennard_jones_potential


sigma = 1
epsilon = 10
r = np.linspace(0.5 * sigma, 3 * sigma, num=1000)
V = lennard_jones_potential(r, sigma, epsilon)

plt.plot(r, V)

r_min = 2 ** (1 / 6) * sigma  # found by solving V'(r)=0
V_min = lennard_jones_potential(r_min, sigma, epsilon)
plt.plot(r_min, V_min, 'o', color='r')
plt.text(r_min, V_min, f"$r_m = {r_min:.4g}$", color='black', va='top')

plt.xlabel(r'$r$')
plt.ylabel(r'$V_{LJ}$')
plt.xlim([0, 3 * sigma])
plt.ylim([-1.5 * epsilon, 3 * epsilon])
plt.grid()
plt.savefig('lennard_jones.png', dpi=200, bbox_inches='tight')
plt.show()
