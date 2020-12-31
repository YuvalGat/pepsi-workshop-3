from dynamics import *

rc = 0.3
r = T0_config(10e-4, 5, 2, rc)
works = True
for p1 in r:
    for p2 in r:
        if p1[0] != p2[0] and p1[1] != p2[1]:
            dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            works = works and (dist > rc)

print(works)
