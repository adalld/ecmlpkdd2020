import time
import subprocess
from src.utils import make_sure_path_exists
from appr.edgelist_to_ligra_format import to_ligra_format
import os


def run_clustering(base_path, dataset, alpha, epsilon):
    t = time.time()

    input_file = os.path.join(base_path, 'data', 'ligra_formats', dataset + '_unweighted_graph')
    if not os.path.isfile(input_file):
        to_ligra_format(base_path=base_path, dataset=dataset)

    output_folder = os.path.join(base_path, 'clusterings', dataset)
    make_sure_path_exists(output_folder)

    run_iter_based_clustering(
        path_to_ligra=os.path.join(base_path, '..', 'ligra'),
        inp_file=input_file,
        epsilon=epsilon,
        alpha=alpha,
        output_folder=output_folder
    )

    return alpha, time.time() - t


def run_iter_based_clustering(path_to_ligra, inp_file, epsilon, alpha, output_folder,
                              first_vertex_to_run_clustering_for=0, last_vertice_to_run_clustering_for=-1):

    path_to_ligra = os.path.join(path_to_ligra, 'apps', 'localAlg', 'ACL-Serial-Opt-Naive')

    basic_call_list = list([
        path_to_ligra,
        '-e', str(epsilon),
        '-rounds', str(0),
        '-first', str(first_vertex_to_run_clustering_for),
        '-last', str(last_vertice_to_run_clustering_for),
    ])

    basic_call_list.append('-s')

    call_list = list(basic_call_list)
    output = os.path.join(output_folder, 'eps_' + '{:1.0e}'.format(float(epsilon)) + '__a_' + '{:1.0e}'.format(float(alpha)))
    call_list.append('-output')
    call_list.append(output)
    call_list.append('-as')
    call_list.append(str(alpha))
    call_list.append(inp_file)
    subprocess.call(call_list)


if __name__ == '__main__':
    run_clustering('/home', 'cora', alpha=0.4, epsilon=1E-04)
