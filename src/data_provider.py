import networkx as nx
import numpy as np
from src.utils import make_sure_path_exists, _LOGGER
from scipy.sparse import csr_matrix
import os
from src.Variables import *


def data_provider(args):
    if args.generate_splits:
        return all_data_generator(args)
    else:
        return load_data_splits_from_txt(args)


def load_data_splits_from_txt(args):
    path = os.path.join(args.base_path, 'data', args.dataset, 'splits')
    if args.instances_per_class:
        path = os.path.join(path, 'instances-per-class', str(args.instances_per_class))
    elif args.training_fraction_per_class:
        path = os.path.join(path, 'training-fraction-per-class', str(int(args.training_fraction_per_class * 100)))
    elif args.training_fraction_classic:
        path = os.path.join(path, 'training-fraction-classic', str(int(args.training_fraction_classic * 100)))
    else:
        raise ValueError('split type is not defined')

    g = nx.read_edgelist(os.path.join(args.base_path, 'data', args.dataset, 'edges.txt'), nodetype=int)
    labels = load_csr(os.path.join(args.base_path, 'data', args.dataset, 'labels.npz')).todense()
    adj = nx.adjacency_matrix(g, nodelist=range(nx.number_of_nodes(g)))

    for i in range(args.num_splits):
        _LOGGER.info('RUNNING ON SPLIT {x}...'.format(x=i + 1))
        _LOGGER.info('Loading split {x} from file...'.format(x=i + 1))
        try:
            masks = np.genfromtxt(os.path.join(path, 'split' + str(i)), dtype=int, delimiter=' ')
        except IOError:
            _LOGGER.info('File {f} not found. You may consider to generate new splits by using the '
                         '--generate_splits flag.'.format(f=os.path.join(path, 'split' + str(i))))
            raise IOError('File {f} not found. You may consider to generate new splits by using the '
                          '--generate_splits flag.'.format(f=os.path.join(path, 'split' + str(i))))
        assert masks.shape[0] == g.number_of_nodes()*3
        train_mask = np.array(masks[:g.number_of_nodes()], dtype=np.bool)
        val_mask = np.array(masks[g.number_of_nodes():2*g.number_of_nodes()], dtype=np.bool)
        test_mask = np.array(masks[2*g.number_of_nodes():], dtype=np.bool)
        assert train_mask.shape[0] == val_mask.shape[0] == test_mask.shape[0]

        y_train, y_val, y_test = np.zeros(labels.shape), np.zeros(labels.shape), np.zeros(labels.shape)
        y_train[train_mask, :] = labels[train_mask, :]
        y_val[val_mask, :] = labels[val_mask, :]
        y_test[test_mask, :] = labels[test_mask, :]

        yield y_train, y_val, y_test, train_mask, val_mask, test_mask, g, adj, labels


def all_data_generator(args):
    g = nx.read_edgelist(os.path.join(args.base_path, 'data', args.dataset, 'edges.txt'), nodetype=int)
    labels = load_csr(os.path.join(args.base_path, 'data', args.dataset, 'labels.npz')).todense()

    if args.instances_per_class:
        for i in range(args.num_splits):
            _LOGGER.info('Generating split {x} using {y} instances per class for training...'
                .format(x=i + 1,
                        y=args.instances_per_class
                        ))
            y_train, y_val, y_test, train_mask, val_mask, test_mask = \
                gen_splits_with_absolute_number_of_instances(labels, args.instances_per_class)

            adj = nx.adjacency_matrix(g, nodelist=range(nx.number_of_nodes(g)))

            to_store = np.hstack((np.hstack((train_mask, val_mask)), test_mask))
            path = os.path.join(args.base_path, 'data', args.dataset, 'splits', 'instances-per-class',
                                str(args.instances_per_class))
            make_sure_path_exists(path)
            np.savetxt(os.path.join(path, 'split' + str(i)), to_store, delimiter=' ', fmt='%i')

            yield y_train, y_val, y_test, train_mask, val_mask, test_mask, g, adj, labels

    elif args.training_fraction_per_class:
        _LOGGER.info('Generating {x} data splits using {y} percent of each class for training and {z} percent of '
                     'instances for test...'
                     .format(x=args.num_splits,
                             y=int(args.training_fraction_per_class * 100),
                             z=int(args.test_fraction * 100)
                             ))
        for i in range(args.num_splits):
            y_train, y_val, y_test, train_mask, val_mask, test_mask = gen_data_splits(
                labels, training_percentage=args.training_fraction_per_class, test_percentage=args.test_fraction)

            adj = nx.adjacency_matrix(g, nodelist=range(nx.number_of_nodes(g)))

            to_store = np.hstack((np.hstack((train_mask, val_mask)), test_mask))
            path = os.path.join(args.base_path, 'data', args.dataset, 'splits', 'training-fraction-per-class',
                                str(int(args.training_fraction_per_class * 100)))
            make_sure_path_exists(path)
            np.savetxt(os.path.join(path, 'split' + str(i)), to_store, delimiter=' ', fmt='%i')

            yield y_train, y_val, y_test, train_mask, val_mask, test_mask, g, adj, labels

    elif args.training_fraction_classic:
        _LOGGER.info('Generating {x} data splits using {y} percent of instances for training and {z} percent of instances '
                     'for test...'
                     .format(x=args.num_splits,
                             y=int(args.training_fraction_classic * 100),
                             z=int(args.test_fraction * 100)
                             ))
        all_y_train, all_y_val, all_y_test, all_train_mask, all_val_mask, all_test_mask = gen_fraction_based_data_splits(
            labels, training_percentage=args.training_fraction_classic, test_percentage=args.test_fraction)

        adj = nx.adjacency_matrix(g, nodelist=range(nx.number_of_nodes(g)))

        for i in range(len(all_test_mask)):
            y_train = all_y_train[i]
            y_val = all_y_val[i]
            y_test = all_y_test[i]
            train_mask = all_train_mask[i]
            val_mask = all_val_mask[i]
            test_mask = all_test_mask[i]

            to_store = np.hstack((np.hstack((train_mask, val_mask)), test_mask))
            path = os.path.join(args.base_path, 'data', args.dataset, 'splits', 'training-fraction-classic',
                                str(int(args.training_fraction_classic * 100)))
            make_sure_path_exists(path)
            np.savetxt(os.path.join(path, 'split' + str(i)), to_store, delimiter=' ', fmt='%i')

            yield y_train, y_val, y_test, train_mask, val_mask, test_mask, g, adj, labels


def gen_splits_with_absolute_number_of_instances(labels, num_train, num_test=1000, num_val=500):
    # TRAIN
    # get <num_train> randomly selected examples for each class
    row_indices_for_classes = []
    for i in range(labels.shape[1]):
        target = np.zeros(labels.shape[1])
        target[i] = 1
        row_indices_for_classes.append(np.where(np.all(labels == target, axis=1))[0])
    training_indices = np.hstack(np.array([np.random.choice(row_indices_for_classes[i], num_train, replace=False)
                         for i in range(len(row_indices_for_classes))]))

    train_mask = sample_mask(training_indices, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]

    # TEST
    # randomly select <num_test> test instances disjoint from training instances
    valid_indices = np.delete(np.arange(0, labels.shape[0]), training_indices)
    test_indices = np.random.choice(valid_indices, num_test, replace=False)
    assert len(list(np.intersect1d(training_indices, test_indices))) == 0

    test_mask = sample_mask(test_indices, labels.shape[0])

    y_test = np.zeros(labels.shape)
    y_test[test_mask, :] = labels[test_mask, :]

    # VALIDATION
    # randomly select <num_val> validation instances disjoint from test and training instances
    valid_indices = np.delete(np.arange(0, labels.shape[0]), np.union1d(training_indices, test_indices))
    val_indices = np.random.choice(valid_indices, num_val, replace=False)
    assert len(list(np.intersect1d(val_indices, test_indices))) == 0
    assert len(list(np.intersect1d(training_indices, val_indices))) == 0
    assert len(list(np.union1d(np.union1d(training_indices, val_indices), test_indices))) == \
           labels.shape[1]*num_train+num_test+num_val
    val_mask = sample_mask(val_indices, labels.shape[0])

    y_val = np.zeros(labels.shape)
    y_val[val_mask, :] = labels[val_mask, :]

    return y_train, y_val, y_test, train_mask, val_mask, test_mask


def gen_data_splits(labels, training_percentage=.7, validation_percentage=.1, test_percentage=.2):
    if training_percentage + validation_percentage + test_percentage > 1.0 + 1e-8:
        raise Exception("Train, val and test percentages are more than one.")

    # get <training_percentage> randomly selected examples for each class
    row_indices_for_classes = []
    for i in range(labels.shape[1]):
        row_indices_for_classes.append(np.where(labels[:, i] == 1)[0])
    training_indices = np.hstack(np.array(
        [np.random.choice(row_indices_for_classes[i],
                          max(1, int(training_percentage * row_indices_for_classes[i].shape[0])),
                          replace=False)
         for i in range(len(row_indices_for_classes))]))

    train_mask = sample_mask(training_indices, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]

    # TEST
    # randomly select <num_test> test instances disjoint from training instances
    valid_indices = np.delete(np.arange(0, labels.shape[0]), training_indices)
    test_indices = np.random.choice(valid_indices, int(test_percentage*labels.shape[0]), replace=False)
    assert len(list(np.intersect1d(training_indices, test_indices))) == 0

    test_mask = sample_mask(test_indices, labels.shape[0])

    y_test = np.zeros(labels.shape)
    y_test[test_mask, :] = labels[test_mask, :]

    # VALIDATION
    # randomly select <num_val> validation instances disjoint from test and training instances
    valid_indices = np.delete(np.arange(0, labels.shape[0]), np.union1d(training_indices, test_indices))
    val_indices = np.random.choice(valid_indices, int(validation_percentage*labels.shape[0]), replace=False)
    assert len(list(np.intersect1d(val_indices, test_indices))) == 0
    assert len(list(np.intersect1d(training_indices, val_indices))) == 0

    val_mask = sample_mask(val_indices, labels.shape[0])

    y_val = np.zeros(labels.shape)
    y_val[val_mask, :] = labels[val_mask, :]

    return y_train, y_val, y_test, train_mask, val_mask, test_mask


def gen_fraction_based_data_splits(labels, n_splits=10, training_percentage=.1, test_percentage=.1,
                                   validation_percentage=.1):
    if training_percentage + validation_percentage + test_percentage > 1.0 + 1e-8:
        raise Exception("Train, val and test percentages are more than one.")

    number_of_nodes = labels.shape[0]
    num_train = int(training_percentage * number_of_nodes)
    num_val = int(validation_percentage * number_of_nodes)
    num_test = int(test_percentage * number_of_nodes)

    y_train, y_val, y_test, train_mask, val_mask, test_mask = [], [], [], [], [], []
    for _ in range(n_splits):
        y_train_, y_val_, y_test_, train_mask_, val_mask_, test_mask_ = do_splits(number_of_nodes, labels, num_train,
                                                                                  num_val, num_test)

        y_train.append(y_train_)
        y_val.append(y_val_)
        y_test.append(y_test_)
        train_mask.append(train_mask_)
        val_mask.append(val_mask_)
        test_mask.append(test_mask_)

    return y_train, y_val, y_test, train_mask, val_mask, test_mask


def do_splits(number_of_nodes, labels, num_train, num_val, num_test):
    node_index = [i for i in range(number_of_nodes)]

    # TRAIN
    train_index = np.random.choice(node_index, num_train, replace=False)
    train_mask = sample_mask(train_index, number_of_nodes)
    y_train = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]

    # VALIDATION
    node_index = list(set(node_index) - set(train_index))
    val_index = np.random.choice(node_index, num_val, replace=False)
    val_mask = sample_mask(val_index, number_of_nodes)
    y_val = np.zeros(labels.shape)
    y_val[val_mask, :] = labels[val_mask, :]

    # TEST
    node_index = list(set(node_index) - set(val_index))
    test_index = np.random.choice(node_index, num_test, replace=False)
    test_mask = sample_mask(test_index, number_of_nodes)
    y_test = np.zeros(labels.shape)
    y_test[test_mask, :] = labels[test_mask, :]

    return y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_csr(path):
    loader = np.load(path)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'], dtype='float64')


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def prepare_train_dev_test_split(y_train, y_val, y_test, train_mask, val_mask, test_mask):
    train_dev_test_split_data = dict()
    train_dev_test_split_data[K_Y_TRAIN] = y_train[train_mask].astype(np.int32)
    train_dev_test_split_data[K_Y_DEV] = y_val[val_mask].astype(np.int32)
    train_dev_test_split_data[K_Y_TEST] = y_test[test_mask].astype(np.int32)
    y_all = np.zeros(y_train.shape, dtype=np.int32)
    y_all[train_mask] = train_dev_test_split_data[K_Y_TRAIN]
    y_all[test_mask] = train_dev_test_split_data[K_Y_TEST]
    y_all[val_mask] = train_dev_test_split_data[K_Y_DEV]
    train_dev_test_split_data[K_Y_ALL] = y_all
    return train_dev_test_split_data
