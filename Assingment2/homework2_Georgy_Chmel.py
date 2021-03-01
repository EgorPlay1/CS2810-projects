import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)


def scatter_data(x, label):
    """ a scatter plot of fish

    Args:
        x (np.array): (n_fish, 2) float, features of each sample
        label (np.array): (n_fish) boolean.  describes which rows of x
            correspond to which group.  (if label[idx] = 0 the this row
            corresponds to group 0)
    """
    plt.scatter(x[label == 0, 1], x[label == 0, 2], label='group 0', color='b')
    plt.scatter(x[label == 1, 1], x[label == 1, 2], label='group 1', color='r')
    plt.legend()


def plot_w(w, x):
    m = -w[1] / w[2]
    b = -w[0] / w[2]

    # a column vector of feature 1 for every sample
    feat_1 = x[:, 1]
    feat_2 = x[:, 2]

    x_boundary = np.linspace(feat_1.min(), feat_1.max(), 5)
    y_boundary = m * x_boundary + b

    plt.plot(x_boundary, y_boundary, color='k', linewidth=2, label='perceptron (w)')
    plt.ylim(feat_2.min(), feat_2.max())
    plt.legend()


def update_perceptron(x, label, w):
    """ updates perceptron by updating once for each sample

    Args:
         x (np.array): (n_sample, n_feature) features
         label (np.array): (n_sample), boolean class label
         w (np.array): (n_feature) initial perceptron weights

    Returns:
        w (np.array): n_features weight vector, defines Linear
            Perceptron
    """
    n_sample, n_feature = x.shape

    for idx_sample in range(n_sample):
        # _x are the features of a single fish
        _x = x[idx_sample, :]

        # _label is the label, 0 or 1, of _x
        _label = label[idx_sample]

        if np.dot(_x, w) < 0 and _label:
            # perceptron estimate: type 0, actual fish type: 1
            w = w + _x
        elif np.dot(_x, w) >= 0 and not _label:
            # perceptron estimate: type 1, actual fish type: 0
            w = w - _x

    return w


def get_data(n_per_grp, mean_grp_0, mean_grp_1):
    """ builds synthetic data

    NOTE: this data already has a constant 1 as its first feature

    Args:
        n_per_grp (int): number of observations, per group, to sample

    Returns:
        x (np.array): (n_sample, 3) float, features of each sample, from
            both groups.  Each row represents one sample (e.g. a fish).
        label (np.array): (n_fish) boolean.  describes which rows of x
            correspond to which group
    """
    # initalize random seed (we get the 'same' random values each time)
    np.random.seed(0)

    # covariance of each group (same cov)
    cov = np.array([[1, -1],
                    [-1, 2]])

    # sample observations from each group
    x_0 = np.random.multivariate_normal(size=n_per_grp,
                                        mean=mean_grp_0,
                                        cov=cov)
    x_1 = np.random.multivariate_normal(size=n_per_grp,
                                        mean=mean_grp_1,
                                        cov=cov)
    # combine samples
    x = np.concatenate((x_0, x_1), axis=0)

    # build label vector
    n_sample = x.shape[0]
    label = np.ones(n_sample, dtype=bool)
    label[:n_per_grp] = False

    # append column of ones (see 'Adding a y-intercept to the Linear Perceptron')
    n_sample = x.shape[0]
    constant_one_col = np.ones((n_sample, 1))
    x = np.concatenate((constant_one_col, x), axis=1)

    return x, label\

x_easy, label_easy = get_data(n_per_grp=100,
                    mean_grp_0 =np.array([0, 0]),
                    mean_grp_1=np.array([5, 5]))

#scatter_data(x_easy, label_easy)

x_hard, label_hard = get_data(n_per_grp=100,
                    mean_grp_0 =np.array([0, 0]),
                    mean_grp_1=np.array([1, 1]))

#scatter_data(x_hard, label_hard)



part_a = """The w will stop changing because it reached a final state where it is 
exactly between the two data sets and is in its final position. Because the line is 
between both data sets and all of the points are on the correct sides the any further
training will not impact the w."""


def stop_angle(w, w_old):
    """ return True when w and w_old are less than 0.05 radians apart

    Args:
        w (np.array): (n_feature) Perceptron Weights after update
        w_old (np.array): (n_feature) Perceptron Weights before update

    Returns:
        stop (bool): boolean, True indicates that training should stop
    """
    if (np.sqrt(np.dot(w, w)) * np.sqrt(np.dot(w_old, w_old))) == 0:
        return False
    else:
        if np.arccos((np.dot(w, w_old))/(np.sqrt(np.dot(w, w)) * np.sqrt(np.dot(w_old, w_old)))) < 0.05:
            return True
        else:
            return False

w = np.array([0, 0, 0])

idx_epoch = 0

# we include a maximum number of iterations
max_epoch = 30

# runs forever, unless we stop it
while True:
    w_old = w
    w = update_perceptron(x_hard, label_hard, w)

    print(f'\n ---epoch {idx_epoch}---')
    print(f'w_old: {w_old}')
    print(f'w: {w}')

    if stop_angle(w_old, w):
        print('Stop Training')
        break

    idx_epoch = idx_epoch + 1

    if idx_epoch >= max_epoch:
        print(f'max_epoch ({max_epoch}) reached: Hard stop')
        break

    print('Keep Training')

scatter_data(x_hard, label_hard)
plot_w(w, x_hard)

part_c = """If we choose an angle that is too low we will be wasting ssystem resourses on nearly useless training that
will take too much time without any real improvenment, or in some cases might even go untill the epoch threshold
If we choose tha angle that is too big we won't give enough training time to the percepatron and the final line won't be
the most efficient line to separate the datasets(has the least amount of false positives) and won't be able to fine tune
to the edge cases"""


from math import sin, cos, pi

A = np.array([[1, 0, 0],
              [0, 2, 0],
              [0, 0, 100]])

B = np.array([[cos(pi/4), -sin(pi/4)],
              [sin(pi/4), cos(pi/4)]])


# these test cases are provided to ensure your work is correct.
# they will throw errors if your solution is wrong.
# uncomment to use them
"""
x = np.array([1, 1, 1])
Ax_expected = np.array([1, 2, 100])
np.testing.assert_array_almost_equal(A @ x, Ax_expected)

x = np.array([1, 1])
Bx_expected = np.array([0, np.sqrt(2)])
np.testing.assert_array_almost_equal(B @ x, Bx_expected)
"""

part_e = """The perceptron is focusing only on one feature at a time because one of the
features is scaled to be 5 orders of magnitude greater that the other feature and its effects on the 
perceptron are so great the impact on the other feature is negligible"""

part_f = """you need to scale the feature to be of the same order of magnitude for the 
percepatron to evaluate them equally"""

x, label = get_data(n_per_grp=100,
                    mean_grp_0 =np.array([0, 0]),
                    mean_grp_1=np.array([5, 5]))
x[: ,1] = x[:, 1] * 0.00001 #divide by 1 million to make them of the same order of magnitude
x[: ,2] = x[:, 2] * 1


plt.show(block=True)
plt.interactive(False)