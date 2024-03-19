import time

import networkx as nx
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def main():
    graph = nx.star_graph(100)

    corr = nx.degree_pearson_correlation_coefficient(graph)
    print(corr)
    plot_distribution_loglog(graph, 'Star Graph', 1)
    plt.show()


def plot_distribution_loglog(graph, title, order):
    df = pd.DataFrame(graph.degree, columns=['Node', 'Degree'])
    dfb = df.groupby('Degree')['Degree'].count().reset_index(name='counts')
    dfb['counts'] = dfb['counts'].div(dfb['counts'].sum())
    dfb = dfb.rename(columns={'counts': 'Probability'})
    print('Degree probability distribution:\n\n', dfb)

    dfb.plot.scatter(x='Degree', y='Probability', title=f'{title} - Degree probability distribution', color=[np.random.rand(3) for i in range(0, len(dfb))])
    plt.yscale('log')
    plt.xscale('log')
    bla, _ = curve_fit(lambda t,b: np.exp(b*t), np.log(dfb['Degree']), dfb['Probability'])
    m = bla
    #m, b = np.polyfit(np.log(dfb['Degree']), np.log(dfb['Probability']), 1)
    plt.plot(dfb['Degree'], np.exp(m*np.log(dfb['Degree'])), ':', color=[0, 0, 0])

    print('\nNetwork Degree Exponent (Approx.): ', -1*m)


if __name__ == '__main__':
    main()