import time
from datetime import datetime
import random

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

MAX_CONNECTIONS = 501
ITERATIONS = 50

def compute_graph():
    nodes = range(1, 1001)
    counts = [0 for node in nodes]
    stop_criteria = False
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    all_links = [(node1, node2) for node1 in nodes for node2 in range(node1+1, 1001)]
    print(len(all_links))

    # random.seed(datetime.now())
    while not stop_criteria:
        link = random.choice(all_links)

        graph.add_edge(link[0], link[1])

        counts[link[0] - 1] = len(nx.node_connected_component(graph, link[0]))
        counts[link[1] - 1] = len(nx.node_connected_component(graph, link[1]))

        print(f"Number of links: {graph.number_of_edges()}. Max occurences value: {counts.index(max(counts)) + 1}. Occurences: {max(counts)}")

        stop_criteria = max(counts) >= MAX_CONNECTIONS

    print(graph.edges)
    print(nx.node_connected_component(graph, counts.index(max(counts)) + 1))

    return graph

def compute_mean_degree(graph):
    df = pd.DataFrame(graph.degree, columns=['Node', 'Degree'])
    mean_degree = df['Degree'].mean()
    print('Graph degree mean:', mean_degree)

    return mean_degree

def plot_mean_degree_distribution(mean_degrees):
    df = pd.DataFrame(mean_degrees, columns=['Mean_Degree'])

    dfb = df.groupby('Mean_Degree')['Mean_Degree'].count().reset_index(name='counts')
    dfb['counts'] = dfb['counts'].div(dfb['counts'].sum())
    dfb = dfb.rename(columns={'counts': 'Probability'})

    print('Degree probability distribution:\n', dfb)
    dfb.plot(kind='bar', x='Mean_Degree', y='Probability', title='Graph mean degree probability distribution')

    plt.show()


def main():
    init_total_time = time.time_ns()
    mean_degrees = []

    for i in range(0, ITERATIONS):
        init_time = time.time_ns()
        print(f"RANDOM GRAPH - ITERATION {i} - BEGIN")
        graph = compute_graph()
        mean_degree = compute_mean_degree(graph)
        mean_degrees.append(mean_degree)
        print(f"RANDOM GRAPH - ITERATION {i} - END. Time : {(time.time_ns() - init_time)/1000000000} Seconds")

    print(f"Total Time : {(time.time_ns() - init_total_time)/1000000000} Seconds")
    plot_mean_degree_distribution(mean_degrees)

if __name__ == '__main__':
    main()