import networkx as nx
import pickle
from src.utils import make_sure_path_exists
import os


def to_ligra_format(base_path, dataset, delimiter=' ', weighted=False):
    edges = os.path.join(base_path, 'data', dataset, 'edges.txt')
    output_graph = os.path.join(base_path, 'data', 'ligra_formats', dataset + '_unweighted_graph')
    output_node_mapping = os.path.join(base_path, 'data', 'ligra_formats', dataset + '_node_mapping')
    make_sure_path_exists(os.path.join(base_path, 'data', 'ligra_formats'))

    g = nx.read_edgelist(edges, delimiter=delimiter, data=weighted, nodetype=int)

    n = g.number_of_nodes()
    m = g.number_of_edges() * 2 - len(list(g.selfloop_edges()))

    old_label_to_new_d = {}
    ind_to_node_map = []
    # no need for it but simpler to read
    counter = 0
    for cur_node in g:
        ind_to_node_map.append(cur_node)
        old_label_to_new_d[cur_node] = counter
        counter += 1

    number_of_edges = []
    nodes_on_other_edge_end = []
    for cur_node in ind_to_node_map:
        edgs = g.edges(cur_node, data=False)
        number_of_edges.append(len(edgs))
        cur_node_other_edge_end = [old_label_to_new_d[e[1]] for e in edgs]
        nodes_on_other_edge_end.extend(cur_node_other_edge_end)

    with open(output_graph, 'w') as f:
        f.write('AdjacencyGraph\n')
        f.write(str(n) + '\n')
        f.write(str(m) + '\n')
        cur_offset = 0
        for n_edges in number_of_edges:
            f.write(str(cur_offset) + '\n')
            cur_offset = cur_offset + n_edges
        for cur_edge_other_end in nodes_on_other_edge_end:
            f.write(str(cur_edge_other_end) + '\n')

    pickle.dump(ind_to_node_map, open(output_node_mapping, 'wb'))