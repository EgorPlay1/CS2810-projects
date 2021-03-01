import numpy as np
import matplotlib.pyplot as plt

def f(x):

    """ evaluates f at point x

    Args:
        x (float): point in domain of x

    Returns:
        f (float): f evaluated @ x
    """
    return 1/4 * x ** 4 + 1/3 * x ** 3 - 3 * x ** 2


def df_dx(x):

    """ returns derivative of f at point x

    Args:
        x (float): points in domain of x

    Returns:
        df (float): derivative of f @ x
    """
    return x ** 3 + x ** 2 - 6 * x

x = np.linspace(-4, 4, 100)
y = np.linspace(-30, 60, 100)

plt.plot(x, f(x), label='f(x)')
#plt.plot(x, df_dx(x), label="f '(x)")
plt.plot(x, np.zeros(x.size), label="x", linewidth=3, color="black")
plt.plot(np.zeros(y.size), y, label="y", linewidth=3, color="black")

plt.legend()


def grad_descent_step_1d(x, learn_rate):
    """ take a single hill climbing step on f

    Args:
        x (float): position in domain of f, where step begins
        learn_rate (float): scales how big step to take (larger
            values take bigger steps)

    Returns:
        x_out (float): position in domain of f, where step ends
    """
    return x + -1 * df_dx(x) * learn_rate


x = 0.1
lr = 0.2
xr = np.array([0.1])
yr = np.array([f(x)])
for i in range(4):
    x = grad_descent_step_1d(x, lr)
    print(x, i+1)
    xr = np.vstack((xr, x))
    yr = np.vstack((yr, f(x)))

plt.scatter(xr, yr)


'''


x0 = np.linspace(0, 5, 20)
x1 = np.linspace(0, 5, 20)


def plot_g_contour(x0, x1):
    X0, X1 = np.meshgrid(x0, x1)
    G = (2 - X0) ** 2 + (3 - X1) ** 2
    cs = plt.contour(X0, X1, G, levels=6)
    plt.gca().clabel(cs, inline=1, fontsize=10)

'''
def g(x):
    """ evaluates g at point x

    Args:
        x (np.array): a 2d point in domain of g

    Returns:
        g (float): evaluation of g
    """
    return (2 - x[0]) ** 2 + (3 - x[1]) ** 2


def del_g(x):
    """ returns gradient of f at point x

    Args:
        x (np.array): 2d point in domain of g

    Returns:
        del_g (np.array): gradient of g at point x (a 2d vector)
    """
    return np.array([2 * (2 - x[0]) * -1, 2 * (3 - x[1]) * -1])


def grad_descent_step_2d(x, learn_rate):
    """ take a single hill climbing step on g

    Args:
        x (np.array): position in domain of g, where step begins (2d vector)
        learn_rate (float): scales how big step to take

    Returns:
        x_out (np.array): position in domain of g, where step ends
    """
    return x + -1 * del_g(x) * learn_rate

'''
plot_g_contour(x0, x1)

x = np.array([1, 1])
lr = 0.2
r = np.array([1, 1])
for i in range(4):
    x = grad_descent_step_2d(x, lr)
    print(x, i+1)
    r = np.vstack((r, x))

plt.scatter(r[:, 0], r[:, 1])
#print(r)
'''
plt.show()
