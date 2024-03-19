import math
import random
import sys
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from networkx.utils import py_random_state
from scipy.optimize import curve_fit


def time_step(t):
    return math.floor(math.sqrt(t))


def closest(lst, K):
    lst = np.asarray(lst)
    idx = (np.abs(lst - K)).argmin()
    return idx

def choose_link(degree_nodes):
    total = 0
    for _, degree in degree_nodes:
        total += degree
    degree_probability = [(node, degree / total) for node, degree in degree_nodes]
    print(f'Probability of chosen node: {degree_probability}')
    choose = random.random()
    choose_sum_probs = 0
    for i in range(0, len(degree_nodes)):
        if choose <= degree_probability[i][1] + choose_sum_probs:
            print(f'Chosen node: {degree_nodes[i][0]}')
            return degree_nodes[i][0]

        choose_sum_probs += degree_probability[i][1]



@py_random_state(2)
def graph_with_accelerated_growth(n, max_time, seed=None):
    source = 1
    # Start with a network with just one node at time t = 0.
    G = nx.Graph()
    G.add_node(source)

    # Repeat until you reach N = 10000.
    while source < n:
        source += 1
        # Then, at each time step t ≥ 1,
        m = time_step(random.random() * (max_time - 1) + 1)
        # add one new node to the network
        G.add_node(source)
        print(f'Number of nodes: {source}')

        targets = []
        # print(G.nodes, G.degree())
        # with m(t) = time_step(t) links to previous nodes.
        degree_previous_nodes = list(G.degree())[0:-1]
        # print(degree_previous_nodes)
        for i in range(0,m):
            # The other extreme of each link will be chosen by preferential attachment, that is,
            # with a probability proportional to the degree of each node (not considering the new links).
            chosen = choose_link(degree_previous_nodes) if source > 2 else 1
            # This may produce multiple edges. Let them be.
            targets.append((source, chosen))

        G.add_edges_from(targets)

    return G

def main():
    # If parameter is passed, limit value of t in case of performance issues/time constraints
    # Else, the value of t is maximum determined by python
    max_t = int(sys.argv[1]) if len(sys.argv) > 1 else sys.maxsize

    graph = graph_with_accelerated_growth(10000, max_t, seed=int(time.time()))

    plot_distribution_loglog(graph, 'Accelerated Growth', 1)
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
    exponent, _ = curve_fit(lambda t,b: np.exp(b*t), np.log(dfb['Degree']), dfb['Probability'])
    plt.plot(dfb['Degree'], np.exp(exponent*np.log(dfb['Degree'])), ':', color=[0, 0, 0])

    degree_exponent = -1 * exponent
    print('\nNetwork Degree Exponent (Approx.): ', degree_exponent)

    # book_statement = 3 + 2*theta/(1-theta) = 3 + 2*0.5/(1-0.5) = 3 + 1/0.5 = 5
    # rate_equation = 1 + 2/(1+theta) = 1 + 2/(1+0.5) = 1 + 2/1.5 = 7/3 = 2.33333333333333
    book_statement = 5
    rate_equation = 7.0/3.0
    if closest([book_statement, rate_equation], degree_exponent) == 0:
        print(f'Book statement has the most successful approach, with {abs(degree_exponent - book_statement)} of delta')
    else:
        print(f'Rate equation has the most successful approach, with {abs(degree_exponent - rate_equation)} of delta')

if __name__ == '__main__':
    main()