import numpy as np

A = np.array([[0.1, 0.7, 0.1],
              [0.2, 0.1, 0.3],
              [0.7, 0.2, 0.6]])

B = np.array([[0.2, 0.3, 0.5],
              [0.4, 0.4, 0.2],
              [0.4, 0.3, 0.3]])

c = np.array([[1],
              [0],
              [0]])

d = np.array([[2],
              [0],
              [0]])

e = np.array([[0.3],
              [0.4],
              [0.3]])

f = np.array([[1],
              [1],
              [1]])

def matmult_n(X, y, n):
    """ left multiplies a vector y, by X, a total of n times

    Args:
        X (np.array): 2d matrix with shape (d, d)
        y (np.array): 1d vector with shape (d, )
        n (int): a positive integer

    Returns:
        Xny (np.array): 1d vector with shape (d, )
    """
    for i in range (n):
        y = np.matmul(X, y)
    return y

print(matmult_n(A, c, 100))
print(matmult_n(A, d, 100))
print(matmult_n(A, e, 100))
print(matmult_n(A, f, 100))
print(matmult_n(B, c, 100))
print(matmult_n(B, d, 100))
print(matmult_n(B, e, 100))
print(matmult_n(B, f, 100))

part_c = """All of the vectors are going in the same direction, but they have different magnitudes. The magnitudes of 
the vetors are dependent on the lengths of the original vectors and are equal to 1 unit for c and e, 2 units for d, and
3 units for f"""