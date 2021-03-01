import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


part_a = """The A matrix is producing a 2D output space and matrix B is producing a 
1D line. We could have predicted it from looking at the span of the vectors in a and in b.
vectors in a were linearly independent and their span was 2D, when vectors in b were
linearly dependent and their span was 1D. From this we can see that the dimension of the solution
is equal to the dimension of the span of the original vectors (i.e. column vectors of a matrix)"""


part_b = """Similarly to part_a, the output of the matrix C is 3D, and the output of matrix D
is 2D. We could have predicted it from looking at the span of the column vectors of the original 
matrices. In C, all 3 column vectors are linearly independent and therefore their span is
3D and their output is also 3D. However in D, 2 of the vectors are linearly dependent and the 
third one is linearly independent to them, therefore their span is 2D which makes the output
be only a 2D plane."""

#C.1

E = np.array([[1, 0, 0],
              [0, 1, 2],
              [0, 0, 0]])

#C.2

E_kernel_basis = np.array([[0],
                           [-1],
                           [2]])

#C.3

E_col_span_basis = np.array([[1, 0],
                             [0, 1]])

#C.4

E_kernel = plt.figure()
ax = E_kernel.add_subplot(111, projection='3d')


# xyz will be a (1000, 3) array, each row is a sample
# we sample a grid of points, something like (0,0), (1,0), (0,1), (1,1)
# see doc on meshgrid for the heart of whats going on here
# check out the other numpy manipulations (linspace, flatten and vstack)
# look to the blue samples below to see what xyz looks like
lim = (-3, 3)
val = np.linspace(lim[0], lim[1], 5)
x, y, z = np.meshgrid(val, val, val)
xyz = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

for sample_idx in range(xyz.shape[0]):
    # choose an alpha vector
    alpha = xyz[sample_idx, :]

    # compute the linear combination of columns of a, b using alpha as weights
    e_alpha = E @ alpha

    # plot them
    h_e = ax.scatter(e_alpha[0], e_alpha[1], e_alpha[2], color='r')

plt.plot([0, 0], [0, -1], [0, 2], linewidth=3)

E_col_span = plt.figure()
ax = E_col_span.add_subplot(111, projection='3d')


# xyz will be a (1000, 3) array, each row is a sample
# we sample a grid of points, something like (0,0), (1,0), (0,1), (1,1)
# see doc on meshgrid for the heart of whats going on here
# check out the other numpy manipulations (linspace, flatten and vstack)
# look to the blue samples below to see what xyz looks like
lim = (-3, 3)
val = np.linspace(lim[0], lim[1], 5)
x, y, z = np.meshgrid(val, val, val)
xyz = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

for sample_idx in range(xyz.shape[0]):
    # choose an alpha vector
    alpha = xyz[sample_idx, :]

    # compute the linear combination of columns of a, b using alpha as weights
    e_alpha = E @ alpha

    # plot them
    h_e = ax.scatter(e_alpha[0], e_alpha[1], e_alpha[2], color='g')

plt.plot([0, 1], [0, 0], linewidth=3)
plt.plot([0, 0], [0, 1], linewidth=3)

plt.show(block=True)