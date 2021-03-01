import numpy as np

g = np.array([[1],
              [1],
              [0],
              [1],
              [0]])

t = np.array([[0],
              [0],
              [1],
              [0],
              [1]])

A = np.array([[2, 1, 1],
              [1, 2, 1],
              [1, 1, 2]])

w, v = np.linalg.eig(A)


print(w)
print(v)