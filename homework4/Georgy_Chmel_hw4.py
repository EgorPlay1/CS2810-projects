import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import pandas as pd
from collections import Counter

# fish_data.csv must be in same folder as this project
df = pd.read_csv('./fish_data.csv')

print(df)
print(Counter(df['Species']))


def scatter_fit_fish(species, a_feat, b_feat, df):
    """ scatters fish data points, fits a straight line between them

    model:

    b = x0 + x1 * a

    Args:
        species (str): specifies which fish data to use ('Perch', 'Bream',
            'Roach', 'Pike', 'Smelt', 'Parkki' or 'Whitefish')
        a_feat (str): which feature to plot along x axis
            (Weight  Length1  Length2  Length3   Height   Width)
        b_feat (str): which feature to plot along y axis
            (Weight  Length1  Length2  Length3   Height   Width)
        df (pd.DataFrame): the pandas dataframe which contains the data
            from all fish (as loaded above)

    Returns:
        x (np.array): parameters of line of best fit (see model above)
    """
    # build a new datafrom from only those rows which describe species
    df_species = df[df['Species'] == species]

    # build a2, b which correspond to a_feat and b_feat
    a2 = df_species[a_feat]
    b = df_species[b_feat]

    # build matrix corresponding to our linear model
    ones = np.ones(a2.size)
    A = np.vstack((ones, a2)).T

    # find p, the projection of b into the column space of A
    p_0 = np.transpose(A) @ b
    p_1 = np.linalg.inv(np.transpose(A) @ A)
    p_2 = p_1 @ p_0
    p = A @ p_2

    # find x, the parameters of the line
    x_0 = np.transpose(A) @ b
    x_1 = np.linalg.inv(np.transpose(A) @ A)
    x = p_1 @ p_0

    # scatter the observed fish
    plt.scatter(a2, b, alpha=.5, label=species)

    # plot a line of the projection
    a2_domain = np.linspace(a2.min(), a2.max(), 100)
    plt.plot(a2_domain, x[0] + x[1] * a2_domain, alpha=.5, linewidth=3)

    # this implementation takes advantage of the fact that each call to
    # plot or scatter cycles through the same set of colors.  So the first
    # call to scatter is blue and the first call to plot is blue (which is
    # important because they're the same fish species)

    # graph grooming
    plt.xlabel(a_feat)
    plt.ylabel(b_feat)
    plt.legend()

    # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5));

    return x

# define features of interest
a_feat = 'Height'
b_feat = 'Weight'

# figsize allows us to make a bigger image
plt.figure(figsize=[20, 10])
x_bream = scatter_fit_fish('Bream', a_feat, b_feat, df)
x_perch = scatter_fit_fish('Perch', a_feat, b_feat, df)
x_pike = scatter_fit_fish('Pike', a_feat, b_feat, df)


perch10 = np.dot(x_perch, np.array([[1], [10]]))
perch_record = np.dot(x_perch, np.array([[1], [17.5]]))
part_b = """a weight of a normal perch with height 10 has a weight = perch10
 and a record perch with height of 17.5 has a weight = perch_record"""

part_c = """We would expect the value with height 10 to be more accurate because the model was trained on the data 
that is on both sides of the given value and the trendline is based on fish that are about this size. The data for 
the record perch is less accurate as we haven't trained on any data close to the value we are looking for and the
treandline is less acurate ouside of the values that it was trained on"""


def scatter_fit_fish_exp(species, a_feat, b_feat, df):
    """ scatters fish data points, fits an exponential line between them

    model:

    b = x0 e^(x1 * a)

    Args:
        species (str): specifies which fish data to use ('Perch', 'Bream',
            'Roach', 'Pike', 'Smelt', 'Parkki' or 'Whitefish')
        a_feat (str): which feature to plot along x axis
            (Weight  Length1  Length2  Length3   Height   Width)
        b_feat (str): which feature to plot along y axis
            (Weight  Length1  Length2  Length3   Height   Width)
        df (pd.DataFrame): the pandas dataframe which contains the data
            from all fish (as loaded above)

    Returns:
        x (np.array): parameters of line of best fit (see model above)
    """
    # build a new datafrom from only those rows which describe species
    df_species = df[df['Species'] == species]

    # build a2, b which correspond to a_feat and b_feat
    a2 = df_species[a_feat]
    b = df_species[b_feat]

    # cast your problem as a linear fit
    b = np.log(b)

    # build matrix corresponding to our linear model
    ones = np.ones(a2.size)
    A = np.vstack((ones, a2)).T

    # find p, the projection of b into the column space of A
    p_0 = np.transpose(A) @ b
    p_1 = np.linalg.inv(np.transpose(A) @ A)
    p_2 = p_1 @ p_0
    p = A @ p_2

    # find x, the parameters of the line
    x_0 = np.transpose(A) @ b
    x_1 = np.linalg.inv(np.transpose(A) @ A)
    x = p_1 @ p_0

    # convert x and b back from 'log space' (for plotting)
    x = np.array([[np.e ** x[0]],
                  [x[1]]])
    b = np.e ** b

    # scatter the observed fish
    plt.scatter(a2, b, alpha=.5, label=species)

    # plot a line of the projection
    a2_domain = np.linspace(a2.min(), a2.max(), 100)
    plt.plot(a2_domain, x[0] * np.exp(x[1] * a2_domain), alpha=.5, linewidth=3)

    # this implementation takes advantage of the fact that each call to
    # plot or scatter cycles through the same set of colors.  So the first
    # call to scatter is blue and the first call to plot is blue (which is
    # important because they're the same fish species)

    # graph grooming
    plt.xlabel(a_feat)
    plt.ylabel(b_feat)
    plt.legend()

    # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5));

    return x

# define features of interest
a_feat = 'Height'
b_feat = 'Weight'

# figsize allows us to make a bigger image
plt.figure(figsize=[20, 10])
x_bream = scatter_fit_fish_exp('Bream', a_feat, b_feat, df)
x_perch = scatter_fit_fish_exp('Perch', a_feat, b_feat, df)
x_pike = scatter_fit_fish_exp('Pike', a_feat, b_feat, df)

plt.show(block=True)