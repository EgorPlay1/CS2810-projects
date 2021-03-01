import numpy as np

X_0 = np.array([[1, 7, 1],
                [2, 1, 3],
                [7, 2, 6]])
X_1 = np.array([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])
X_2 = np.array([[4, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])

def martix_is_solved(X):
    if X[0, 0] == 1 and X[1, 0] == 0 and X[2, 0] == 0:
        if X[1, 1] == 1 and X[0, 1] == 0 and X[2, 1] == 0:
            if X[2, 2] == 1 and X[0, 2] == 0 and X[1, 2] == 0:
                return True
            elif X[2, 2] == 0:
                return True
            else:
                return False
        else:
            return False
    else:
        return False

def multiply_array(X, r, n):
    if r == 1:
        X = np.array([[n*X[0,0], n*X[0, 1], n*X[0, 2]],
                     [X[1, 0], X[1, 1], X[1, 2]],
                     [X[2, 0], X[2, 1], X[2, 2]]])
    elif r == 2:
        X = np.array([[X[0,0], X[0, 1], X[0, 2]],
                     [n*X[1, 0], n*X[1, 1], n*X[1, 2]],
                     [X[2, 0], X[2, 1], X[2, 2]]])
    else:
        X = np.array([[X[0,0], X[0, 1], X[0, 2]],
                     [X[1, 0], X[1, 1], X[1, 2]],
                     [n*X[2, 0], n*X[2, 1], n*X[2, 2]]])
    return X

def Linear_EQ_Solver(X):
    if martix_is_solved(X):
        print(X)
        return X
    elif X[0, 0] != 1:
        X = multiply_array(X, 0, 1/X[0, 0])
        Linear_EQ_Solver(X)
        return X
    else:
        return "error"

print(Linear_EQ_Solver(X_2))