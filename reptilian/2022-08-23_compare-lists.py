import numpy as np
a = [10, 20, 30, 40, 50]
b = [11, 22, 29, 41, 51]
s = [np.sign(x - y) for x, y in zip(a, b)]
# Assumes no 0 values because no token is in more than one block
print(f"{s}")
print(f"{set(s)}")
print(f"{len(set(s))}")
# If count is 1, all signs agree (assuming no 0 values), and
# therefore no block transpositions
