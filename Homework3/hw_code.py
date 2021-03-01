import numpy as np

S = np.array([[4, 2,-1, 5, 2],
              [3, 5, 4, 5, 4],
              [-1, 0, -3, 5, -2],
              [-2, -1, -5, 5, 0],
              [2, 3, 2, -5, -4]])

S_inv = np.linalg.inv(S)

a = np.array([[5], [4], [-3], [0], [4]])
b = np.array([[0], [-2], [4], [-1], [-1]])

print(S_inv @ a)
print(S_inv @ b)