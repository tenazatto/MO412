import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from scipy.optimize import curve_fit

# Adjust for better ratio precision
RATIO_TOLERANCE = 0.000001

def p_infinite(f, fc, beta):
    return np.math.pow(abs(f - fc), beta) if f <= fc else 0

def find_ratio(ratio, f, fc, beta):
    return p_infinite_ratio_binary_search(ratio, f, fc, beta, fc)

# Considering a function as an array of result numbers and ordered indexes,
# and considering that this function is a injective or a bijective function
# we can do a binary search approach with some tolerance to find an approximation
# and didn't need to adapt newton-raphson method to it, also skipping
# some differential calculations and treating the function only with its own calculations
def p_infinite_ratio_binary_search(ratio, f, fc, beta, f_conv):
    f_middle = (f+f_conv)/2
    ratio_p = p_infinite(f_middle, fc, beta)/p_infinite(0, fc, beta)

    if abs(ratio_p-ratio) < RATIO_TOLERANCE:
        return f
    elif ratio_p > ratio:
        return p_infinite_ratio_binary_search(ratio, f_middle, fc, beta, f_conv)
    elif ratio_p < ratio:
        return p_infinite_ratio_binary_search(ratio, f, fc, beta, f_middle)


# 8.C Critical Threshold Under Random Failures
def critical_threshold_under_random_failures(df):
    k = df['Degree'].mean()
    fc = 1 - 1 / ((k * k / k) - 1)
    return fc, k


# 8.A Percolation in Scale-free Network
def scale_free_critical_exponent_beta(gamma):
    beta = 1  # gamma > 4
    if 2 < gamma < 3:
        beta = 1 / (3 - gamma)
    elif 3 < gamma < 4:
        beta = 1 / (gamma - 3)
    return beta


def main():
    graph_data = open('netA.txt', "r")
    graph_type = nx.Graph()

    graph = nx.parse_edgelist(graph_data, comments='t', delimiter='\t', create_using=graph_type,
                          nodetype=int, data=(('weight', float),))

    df = pd.DataFrame(graph.degree, columns=['Node', 'Degree'])

    dfb = df.groupby('Degree')['Degree'].count().reset_index(name='counts')
    dfb['counts'] = dfb['counts'].div(dfb['counts'].sum())
    dfb = dfb.rename(columns={'counts': 'Probability'})

    bla, _ = curve_fit(lambda t,b: np.exp(b*t), np.log(dfb['Degree']), dfb['Probability'])
    m = bla

    gamma = -1*m
    beta = scale_free_critical_exponent_beta(gamma)
    fc, k = critical_threshold_under_random_failures(df)

    print('Graph degree mean:', k)
    print('Network Degree Exponent (Approx.): ', gamma)
    print('Critical threshold: ', fc)

    f = float(0)
    p_infinite_f_zero_ratios = []
    while int(f*100) <= 100:
        # print(k, beta, f, fc, p_infinite(f, fc, beta), p_infinite(f, fc, beta) / p_infinite(0, fc, beta))
        p_infinite_f_zero_ratios.append((f, p_infinite(f, fc, beta) / p_infinite(0, fc, beta)))
        f += 0.05

    dfc = pd.DataFrame(p_infinite_f_zero_ratios, columns=['f', 'P∞( f)/P∞(0)'])

    print('\nDegree probability distribution:\n', dfb)
    print('\nP∞( f)/P∞(0):\n', dfc)

    f_half_ratio = find_ratio(0.5, 0, fc, beta)
    print("\nValue of f when P∞( f)/P∞(0) = 50%:", f_half_ratio)

    matplotlib_axes_logger.setLevel('ERROR')
    dfc.plot.scatter(x='f', y='P∞( f)/P∞(0)', title=f'Robustness against random failures',
                     color=[0.8, 1, 0.2], edgecolors=[0, 1, 0])
    plt.axhline(color='black', lw=0.75)
    plt.axvline(color='black', lw=0.75)
    plt.plot([0, 1], [1, 0], color=[0, 0, 0])

    plt.show()



if __name__ == '__main__':
    main()