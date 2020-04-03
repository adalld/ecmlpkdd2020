import os
import errno
import numpy as np
import scipy.sparse as sp
import logging
from src.Variables import *
import argparse


logging.basicConfig(format='[%(levelname)s] %(asctime)-15s %(message)s', level=logging.INFO)
_LOGGER = logging.getLogger(name=os.path.basename(__file__))


def parse_args():
    #
    # Parses Ada-LLD arguments.
    #
    parser = argparse.ArgumentParser(description="Run Ada-LLD.")
    # General
    parser.add_argument('--dataset', type=str,
                        choices= multiclass_datasets + multilabel_datasets,
                        default='cora',
                        help='Input dataset. Dataset must be in the data folder. Also,'
                             'it must be registered in the corresponding collection in the starter.py script. '
                             'Currently available sets are {d} for multiclass and {l} for multilabel tasks. REQUIRED'
                        .format(d=multiclass_datasets, l=multilabel_datasets))

    parser.add_argument('--models', nargs='+', required=True,
                        choices=models,
                        help='The desired models, must be in {m}. REQUIRED'.format(m=models))

    parser.add_argument('--base_path', type=str, default='../',
                        help='The path pointing to the folder containing the Ada-LLD and ligra project folders.')

    parser.add_argument('--multilabel', action='store_true', default=False,
                        help='If this option is chosen, a multilabel task is considered.')

    parser.add_argument('--weighting_factor', type=int, default=10,
                        help='Only takes effect for multilabel classification tasks. Weights the positive labels, as'
                             'they are likely to be highly underrepresented.')
    # Configuration for data splits
    data_split_mode = parser.add_mutually_exclusive_group()
    data_split_mode.add_argument('--instances_per_class', type=int,
                                 help='Number of instances per class that shall be used for training.')

    data_split_mode.add_argument('--training_fraction_per_class', type=float,
                                 help='Training fraction per class that shall be used for training.')

    data_split_mode.add_argument('--training_fraction_classic', type=float,
                                 help='Training fraction that shall be used for training.')

    parser.add_argument('--test_fraction', type=float, default=0.2,
                        help='Fraction of data used for test.')

    parser.add_argument('--num_splits', type=int, default=10,
                        help='Number of splits.')

    parser.add_argument('--generate_splits', action='store_true', default=False,
                        help='Whether to load data splits from disk or to generate and store new splits.')
    # Learning parameters
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate.')

    parser.add_argument('--epochs', type=int, default=10000,
                        help='Number of epochs.')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='Number of samples per batch.')

    parser.add_argument('--patience', type=int, default=100,
                        help='Patience for early stopping.')

    parser.add_argument('--hidden_dim', type=int, default=16,
                        help='Number of neurons in hidden layer.')

    parser.add_argument('--log_interval', type=int, default=1000,
                        help='Number of epochs after which to print intermediate learning results to console.')
    # APPR parameters
    parser.add_argument('--appr_alphas', nargs='+', type=float,
                        help='Alpha values [0,1) which determine the locality for the appr computation.')

    parser.add_argument('--appr_epsilon', type=float, default=0.00001,
                        help='Approximation threshold. Default is 1E-5.')

    return parser.parse_args()


def check_args(args):
    if args.dataset in multiclass_datasets:
        if args.multilabel:
            _LOGGER.info('The {d} dataset is a multiclass dataset. Changing to multiclass mode.'.format(d=args.dataset))
            args.multilabel = False
            # currently only multilabel datasets require class weight
            args.weighting_factor = 1.0

    elif args.dataset in multilabel_datasets:
        if not args.multilabel:
            _LOGGER.info('The {d} dataset is a multilabel dataset. Changing to multilabel mode. Using weighting factor '
                         '{w} for positive labels.'.format(d=args.dataset, w=args.weighting_factor))
            args.multilabel = True
        if not args.training_fraction_per_class and not args.training_fraction_classic:
            args.training_fraction_per_class = 0.7
            args.instances_per_class = None
            _LOGGER.info('Also, the data split mode is changed to common train-val-test splits rather than using a '
                         'fixed number of training instances per class for multilabel predictions. The training '
                         'fraction is set to {f}.'.format(f=args.training_fraction_per_class))
        if '2step_lp' in args.models:
            _LOGGER.info('The 2-step Label Propagation method is not applicable to multilabel problems and therefore '
                         'will be skipped.')

    else:
        # should never happen
        raise ValueError('The dataset is unknown! Is it registered in the header of the starter.py script? It must '
                         'appear in either one of the _multiclass_datasets or _multilabel_datasets list!')


def make_sure_path_exists(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx
