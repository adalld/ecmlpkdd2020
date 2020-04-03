import numpy as np
import json
import pickle
from scipy.sparse import csr_matrix
import os
from src.utils import _LOGGER
from appr.call_ligra import run_clustering


def compute_label_distributions(dataset, base_path, labels, alphas, epsilon, train_dev_test_masks, normalize=True, sparse_pprs=None):
    train_X, dev_X, test_X = [], [], []
    if not sparse_pprs:
        sparse_pprs = remapped_pprs_as_sparse_matr(base_path, dataset, alphas, epsilon)
        # conformance check to ensure that the own labels won't be considered in any case
        ensure_diags_are_zero(sparse_pprs)
    # preprocess the label matrices to derive the seen labels during training & dev, resp. test, phase
    label_comb = preprocess_label_matrices(csr_matrix(labels), train_dev_test_masks)
    # consider only the labels of the training set for calculating the input representations for the training set
    add_label_distrib_for_each_ppr(label_comb['train_labels'], sparse_pprs, train_X, normalize=normalize)
    # consider only the labels of the training set for calculating the input representations for the dev set
    add_label_distrib_for_each_ppr(label_comb['train_labels'], sparse_pprs, dev_X, normalize=normalize)
    # consider labels of the training and dev set for calculating the input representations for the test set
    add_label_distrib_for_each_ppr(label_comb['train_and_dev_labels'], sparse_pprs, test_X, normalize=normalize)

    return train_X, dev_X, test_X, sparse_pprs


# Might be of interest to parallelize at this point here, e.g., by submitting the load_ppr_with_remapping calls to some
# ProcessPoolExecutor or so
def remapped_pprs_as_sparse_matr(base_path, dataset, alphas, epsilon):
    sparse_pprs = []
    for alpha in alphas:
        ppr_data_dict = load_ppr_with_remapping(base_path, dataset, alpha, epsilon)
        sparse_pprs.append(convert_dict_to_csr(ppr_data_dict))
    return sparse_pprs


def load_ppr_with_remapping(base_path, dataset, alpha, epsilon):
    path = os.path.join(base_path, 'clusterings', dataset, 'eps_%.0e__a_%.0e' % (epsilon, alpha)) # for classic appr
    # if the appr file does not exist, run clustering
    if not os.path.isfile(path):
        _, _ = run_clustering(base_path, dataset, alpha, epsilon)
    # try loading the appr data
    try:
        ppr_data = json.load(open(path, 'r'))
    except FileNotFoundError:
        _LOGGER.info('File {f} was not found!'.format(f=path))
        raise FileNotFoundError('File {f} was not found!'.format(f=path))
    except IOError:
        _LOGGER.info('The program was not able to load the data from file {f}.'.format(f=path))
        raise IOError('The program was not able to load the data from file {f}.'.format(f=path))
    # remap the node ids from the ligra output
    mapping = pickle.load(open(os.path.join(base_path, 'data', 'ligra_formats', dataset + '_node_mapping'),
                               'rb'))
    mapper = {idx: n_id for idx, n_id in enumerate(mapping)}
    remapped_ppr_data = dict()
    for idx, n_id in enumerate(mapping):
        remapped_vec = []
        for i in range(len(ppr_data[str(idx)])):
            remapped_vec.append([mapper[ppr_data[str(idx)][i][0]], ppr_data[str(idx)][i][1]])
        remapped_ppr_data[str(n_id)] = remapped_vec

    return remapped_ppr_data


def convert_dict_to_csr(ppr_data_dict):
    data, row_ind, col_ind = [], [], []
    for node, neigh_prob in ppr_data_dict.items():
        for neigh, prob in neigh_prob:
            row_ind.append(int(node))
            col_ind.append(int(neigh))
            data.append(prob)
    sparse_ppr = csr_matrix( (data, (row_ind, col_ind)), shape=(len(ppr_data_dict), len(ppr_data_dict)) )

    return sparse_ppr


def ensure_diags_are_zero(sparse_pprs):
    for curr_ppr in sparse_pprs:
        assert np.isclose(curr_ppr.diagonal().sum(), 0)


def preprocess_label_matrices(labels, mask_train_dev_test):
    train_mask = mask_train_dev_test[0]
    train_labels = csr_zero_rows(labels.copy(), train_mask)

    train_and_dev_mask = np.logical_or(train_mask, mask_train_dev_test[1])
    train_and_dev_labels = csr_zero_rows(labels.copy(), train_and_dev_mask)

    ret = {
        'train_labels': train_labels,
        'train_and_dev_labels': train_and_dev_labels
    }

    return ret


def add_label_distrib_for_each_ppr(allowed_labels, sparse_pprs, curr_set, normalize):
    for ppr_mat in sparse_pprs:
        label_distributions = compute_label_distributions_for_ppr(ppr_mat, allowed_labels, normalize=normalize)
        curr_set.append(label_distributions)


def compute_label_distributions_for_ppr(ppr_mat, allowed_labels, normalize=True):
    label_distributions = ppr_mat.dot(allowed_labels).todense()
    norm_sum = label_distributions.sum(axis=1)
    mask_nzerovec = np.squeeze(np.asarray(norm_sum > 0.))
    if normalize:
        label_distributions[mask_nzerovec] = label_distributions[mask_nzerovec] / norm_sum[mask_nzerovec]
    return label_distributions


def csr_zero_rows(csr, mask_to_keep):
    mask = np.logical_not(mask_to_keep)
    csr[mask] = 0
    csr.eliminate_zeros()
    return csr

