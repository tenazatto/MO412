import csv

import networkx as nx
import networkx.algorithms.community as nxc


def compare_communities(graph):
    lst_fluid = sorted([sorted(community) for community in nxc.asyn_fluidc(graph, 5)])
    lst_lpa = sorted([sorted(community) for community in nxc.asyn_lpa_communities(graph)])

    print(lst_fluid)
    print(lst_lpa)
    print("Equal communities" if lst_lpa == lst_fluid else "Different communities")

def generate_files(graph):
    lst_fluid = sorted([sorted(community) for community in nxc.asyn_fluidc(graph, 5)])
    lst_lpa = sorted([sorted(community) for community in nxc.asyn_lpa_communities(graph)])

    i = 0
    for community in lst_fluid:
        with open(f"fluid_comm_{i}.txt", "w", encoding='utf8') as data_file:
            for node in community:
                data_file.write(node + '\n')
        i+=1

    i = 0
    for community in lst_lpa:
        with open(f"lpa_comm_{i}.txt", "w", encoding='utf8') as data_file:
            for node in community:
                data_file.write(node + '\n')
        i+=1

def main():
    graph = read_graph_from_file('netM.csv')

    generate_files(graph)

    compare_communities(graph)


def read_graph_from_file(filename):
    nodes = []
    edges = []
    graph = nx.Graph()
    graph_data = open(filename, "r")
    read_csv = csv.reader(graph_data, delimiter=" ")

    for row in read_csv:
        print(row)
        nodes.extend(row)
        for column in range(1, len(row)):
            edges.append((row[0], row[column]))

    graph.add_nodes_from(list(set(nodes)))
    graph.add_edges_from(edges)

    return graph


if __name__ == '__main__':
    main()