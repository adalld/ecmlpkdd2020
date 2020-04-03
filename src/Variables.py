# data_tr_dev_test dictionary variable
TRAIN_DEV_TEST_SPLIT_DATA = 'data_tr_dev_test_split'
# prepare_train_dev_test_split variables
K_Y_TRAIN = 'k_y_train'
K_Y_DEV = 'k_y_dev'
K_Y_TEST = 'k_y_test'
K_Y_ALL = 'k_y_all'

K_R_ACC = 'acc'
K_R_MICROF1 = 'microf1'
K_R_MACROF1 = 'macrof1'

# preprocessed_data dict variables
Y_TRAIN = 'y_train'
Y_VAL = 'y_val'
Y_TEST = 'y_test'
TRAIN_MASK = 'train_mask'
VAL_MASK = 'val_mask'
TEST_MASK = 'test_mask'
G = 'g'
ADJ = 'adj'
PPR = 'ppr'
LABELS = 'labels'

# label distribution split variables
LD_TRAIN_X = 'ld_train_x'
LD_DEV_X = 'ld_dev_x'
LD_TEST_X = 'ld_test_x'

RUN_TIMES = 'run_times'

LP_BETA = 'lp_beta'
LP_ALPHA = 'lp_alpha'

BINARY_CLASSIFICATION_PROBABILITY_THRESHOLD = 0.5

multiclass_datasets = ['cora', 'citeseer', 'pubmed', 'sbm_homophily', 'sbm_heterophily', 'sbm_mix',
                        'sbm_mixed_patterns']
multilabel_datasets = ['blogcatalog', 'imdb_germany']

models = ['ld_avg', 'ld_concat', 'ld_indp', 'ld_shared']

