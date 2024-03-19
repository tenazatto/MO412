import time

import networkx as nx
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def main():
    # graph = nx.barabasi_albert_graph(10000,1, seed=int(time.time()))
    graph = nx.erdos_renyi_graph(100, 0.4)

    compute_mean_degree(graph)
    # plot_distribution_loglog(graph, 'Barabási-Albert Graph', 1)
    # plt.show()


def plot_distribution_loglog(graph, title, order):
    df = pd.DataFrame(graph.degree, columns=['Node', 'Degree'])
    dfb = df.groupby('Degree')['Degree'].count().reset_index(name='counts')
    dfb['counts'] = dfb['counts'].div(dfb['counts'].sum())
    dfb = dfb.rename(columns={'counts': 'Probability'})
    print('Degree probability distribution:\n\n', dfb)

    dfb.plot.scatter(x='Degree', y='Probability', title=f'{title} - Degree probability distribution', color=[np.random.rand(3) for i in range(0, len(dfb))])
    plt.yscale('log')
    plt.xscale('log')
    # bla, _ = curve_fit(lambda t,b: np.exp(b*t), np.log(dfb['Degree']), dfb['Probability'])
    m = bla
    #m, b = np.polyfit(np.log(dfb['Degree']), np.log(dfb['Probability']), 1)
    plt.plot(dfb['Degree'], np.exp(m*np.log(dfb['Degree'])), ':', color=[0, 0, 0])

    print('\nNetwork Degree Exponent (Approx.): ', -1*m)

def compute_mean_degree(graph):
    df = pd.DataFrame(graph.degree, columns=['Node', 'Degree'])
    mean_degree = df['Degree'].mean()
    print('Graph degree mean:', mean_degree)

    return mean_degree

if __name__ == '__main__':
    main()