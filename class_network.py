import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import csv

def main():
    graph, node_colors, hobbies, persons = read_bipartite_graph_from_file('class-network.tsv')

    print('\nBipartite graph\n')
    top = nx.bipartite.sets(graph)[0]
    pos = nx.bipartite_layout(graph, top)
    nx.draw(graph, pos=pos, with_labels=True, node_color=node_colors)
    compute_graph_metrics(graph, 'Bipartite graph')
    nx.write_gexf(graph, "bipartite_graph.gexf")
    plt.show()

    print('\nProjection graph 1\n')
    proj1 = nx.bipartite.weighted_projected_graph(graph, hobbies)
    nx.draw(proj1, with_labels=True, node_color='green')
    compute_graph_metrics(proj1, 'Projection graph 1')
    nx.write_gexf(proj1, "projection_graph_1.gexf")
    plt.show()

    print('\nProjection graph 2\n')
    proj2 = nx.bipartite.weighted_projected_graph(graph, persons)
    nx.draw(proj2, with_labels=True, node_color='blue')
    compute_graph_metrics(proj2, 'Projection graph 2')
    nx.write_gexf(proj2, "projection_graph_2.gexf")
    plt.show()


def read_bipartite_graph_from_file(filename):
    persons = []
    hobbies = []
    node_colors = []
    graph = nx.Graph()
    graph_data = open(filename, "r")
    read_tsv = csv.reader(graph_data, delimiter="\t")

    for row in read_tsv:
        print(row)
        if row[0] == '':
            hobbies = row
            graph.add_nodes_from(row, bipartite=0)
            graph.remove_node(row[0])
            for i in range(1, len(row)):
                node_colors.append('green')
        else:
            graph.add_node(row[0], bipartite=1)
            persons.append(row[0])
            node_colors.append('blue')
            for i in range(1, len(row)):
                if row[i] == '1':
                    graph.add_edge(hobbies[i], row[0])

    hobbies.remove('')

    return graph, node_colors, hobbies, persons


def compute_graph_metrics(graph, title):
    print('Graph clustering coefficients:', nx.clustering(graph, weight=True))
    conn_comp = sorted(nx.connected_components(graph), key=len)
    print('Graph connected components:', conn_comp)
    print('Connected graph' if len(conn_comp) == 1 else 'Disconnected graph')
    print('Graph average clustering coefficient:', nx.average_clustering(graph, weight=True))
    print('Graph average distance:', nx.average_shortest_path_length(graph))
    df = pd.DataFrame(graph.degree, columns=['Node', 'Degree'])
    print('Graph degree mean:', df['Degree'].mean())
    print('Graph degree standard deviation:', df['Degree'].std())
    dfb = df.groupby('Degree')['Degree'].count().reset_index(name='counts')
    dfb['counts'] = dfb['counts'].div(dfb['counts'].sum())
    dfb = dfb.rename(columns={'counts': 'Probability'})
    print('Degree probability distribution:\n', dfb)

    dfb.plot.scatter(x='Degree', y='Probability', title=f'{title} - Degree probability distribution', color=[np.random.rand(3) for i in range(0, len(dfb))])
    plt.yscale('log')
    plt.xscale('log')


if __name__ == '__main__':
    main()