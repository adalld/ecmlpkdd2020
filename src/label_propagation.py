from scipy.sparse import diags
from sklearn.metrics import f1_score
import argparse
from itertools import product

from src.utils import _LOGGER
from src.data_provider import *

'''
2-step LP
https://arxiv.org/pdf/1612.05001.pdf 2 steps label propagation
'''
def two_step_label_propagation(preprocessed_data):
    adj = preprocessed_data[ADJ]
    n, c = preprocessed_data[Y_TRAIN].shape
    degree = np.array(adj.sum(1)).flatten()
    D = diags(np.power(degree + 1e-200, -.5), 0, format='csc')
    Lnorm = D.dot(adj).dot(D)

    Y = np.ones((n, c)) / c
    Y[preprocessed_data[TRAIN_MASK], :] = 0
    class_labels_training = np.argwhere(preprocessed_data[Y_TRAIN][preprocessed_data[TRAIN_MASK]])[:,1]
    Y[preprocessed_data[TRAIN_MASK], class_labels_training] = 1.

    for alpha in preprocessed_data[LP_ALPHA]:
        for beta in preprocessed_data[LP_BETA]:
            F = Y.copy()  # initialise predictions F
            diff = 1
            it = 0
            F_old = -1
            while diff > 1e-6 and it < 200:
                for _ in range(beta):
                    for __ in range( 2):
                        F = Lnorm.T.dot(F)
                F = (1 - alpha) * preprocessed_data[Y_TRAIN] + alpha * F

                diff = np.sum(np.abs(F - F_old))
                it += 1
                F_old = F.copy()

            predicted_labels = np.argmax(F, 1 )
            predicted_labels_matrix = np.zeros((n,c))
            predicted_labels_matrix[np.arange(n),predicted_labels] = 1.

            test_micro_f1, test_macro_f1, _ = skl_f1_score(
                pred=predicted_labels_matrix[preprocessed_data[TEST_MASK]],
                ty=preprocessed_data[Y_TEST][preprocessed_data[TEST_MASK]]
            )

            test_accuracy = np.sum(
                (predicted_labels[preprocessed_data[TEST_MASK]] ==
                 np.argmax(preprocessed_data[Y_TEST][preprocessed_data[TEST_MASK]],1)) /
                float(predicted_labels[preprocessed_data[TEST_MASK]].shape[0]))

            return alpha, beta, test_accuracy


def skl_f1_score(pred, ty):
    return f1_score(ty, y_pred=pred, average='micro'), f1_score(ty, y_pred=pred, average='macro'), None


if __name__ == '__main__':
    #
    # Parses 2step LP arguments.
    #
    parser = argparse.ArgumentParser(description="Run 2step LP.")
    # General
    parser.add_argument('--dataset', type=str, required=True,
                        choices=multiclass_datasets,
                        help='Input dataset. Dataset must be in the data folder. Also,'
                             'it must be registered in the corresponding collection in the starter.py script. '
                             'Currently available sets are {d}. REQUIRED'
                        .format(d=multiclass_datasets))

    parser.add_argument('--base_path', type=str, default='../',
                        help='The path pointing to the folder containing the Ada-LLD and ligra project folders.')

    # Configuration for data splits
    data_split_mode = parser.add_mutually_exclusive_group()
    data_split_mode.add_argument('--instances_per_class', type=int,
                                 help='Number of instances per class that shall be used for training.')

    data_split_mode.add_argument('--training_fraction_per_class', type=float,
                                 help='Training fraction per class that shall be used for training.')

    data_split_mode.add_argument('--training_fraction_classic', type=float,
                                 help='Training fraction that shall be used for training.')

    parser.add_argument('--num_splits', type=int, default=1,
                        help='Number of splits.')

    parser.add_argument('--generate_splits', action='store_true', default=False,
                        help='Whether to load data splits from disk or to generate and store new splits.')
    # APPR parameters
    parser.add_argument('--appr_alphas', nargs='+', type=float,
                        help='Alpha values [0,1) which determine the locality for the appr computation.')

    parser.add_argument('--appr_epsilon', type=float, default=0.0001,
                        help='Approximation threshold. Default is 1E-4.')
    # 2-step LP parameters
    parser.add_argument('--lp_alphas', nargs='+', type=float,
                        help='Alpha values [0,1) which determine the locality for the 2-step label propagation.')

    parser.add_argument('--lp_betas', nargs='+', type=int,
                        help='Beta parameters for the 2-step label propagation.')

    args = parser.parse_args()

    # Generate data splits and write them to disk
    if args.generate_splits:
        _LOGGER.info('Generating {n} data splits...'.format(n=args.num_splits))
        for _, _, _, _, _, _, _, _, _ in data_provider(args):
            pass
        args.generate_splits = False

    i = 1
    for y_train, y_val, y_test, train_mask, val_mask, test_mask, g, adj, labels in data_provider(args):
        print(f'Split {i}:')
        lp_results = {a: {b: {} for b in args.lp_betas} for a in args.lp_alphas}
        for alpha, beta in product(args.lp_alphas, args.lp_betas):
            remaining_mask = np.ones(train_mask.shape[0]) - train_mask - val_mask - test_mask

            preprocessed_data = {
                Y_TRAIN: y_train,
                Y_VAL: y_val,
                Y_TEST: y_test,
                TRAIN_MASK: train_mask,
                VAL_MASK: val_mask,
                TEST_MASK: test_mask,
                G: g,
                ADJ: adj,
                LP_ALPHA: [alpha],
                LP_BETA: [beta]
            }

            # run 2step LP
            alpha, beta, acc = two_step_label_propagation(preprocessed_data)

            print(f'alpha = {alpha}, beta = {beta}: accuracy = {acc}')
        i += 1


