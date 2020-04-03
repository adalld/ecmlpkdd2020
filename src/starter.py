import numpy as np

from src.torch_models.training import Trainer
from src.torch_models.models import *
from src.torch_models.utils import convert_to_torch_input

from src.utils import _LOGGER, parse_args, check_args
from src.data_provider import data_provider
from src.ld_preprocessing import compute_label_distributions
from src.Variables import multilabel_datasets


if __name__ == '__main__':
    args = parse_args()
    check_args(args)

    # Generate data splits and write them to disk
    if args.generate_splits:
        _LOGGER.info('Generating {n} data splits...'.format(n=args.num_splits))
        for _, _, _, _, _, _, _, _, _ in data_provider(args):
            pass
        args.generate_splits = False

    res = {m: [] for m in args.models}
    for y_train, y_val, y_test, train_mask, val_mask, test_mask, g, adj, labels in data_provider(args):

        _LOGGER.info('Pre-computing label distributions...')
        ld_train_x, ld_dev_x, ld_test_x, sparse_pprs = compute_label_distributions(
            dataset=args.dataset,
            base_path=args.base_path,
            labels=labels,
            alphas=args.appr_alphas,
            epsilon=args.appr_epsilon,
            train_dev_test_masks=[train_mask, val_mask, test_mask]
        )

        _LOGGER.info('Starting to train the model(s)...')
        train_x, train_y = convert_to_torch_input(
            list_x=ld_train_x,
            dense_y=y_train,
            mask=train_mask,
            multilabel=args.multilabel
        )
        test_x, test_y = convert_to_torch_input(
            list_x=ld_test_x,
            dense_y=y_test,
            mask=test_mask,
            multilabel=args.multilabel
        )
        val_x, val_y = convert_to_torch_input(
            list_x=ld_dev_x,
            dense_y=y_val,
            mask=val_mask,
            multilabel=args.multilabel
        )

        output = y_train.shape[1]
        for model_str in args.models:

            if model_str == 'ld_avg':
                model = LD_AVG(
                    h1_dim=args.hidden_dim,
                    num_labels=output,
                    num_neighbs=len(ld_train_x)
                )
            elif model_str == 'ld_concat':
                model = LD_CONCAT(
                    h1_dim=args.hidden_dim,
                    num_labels=output,
                    num_neighbs=len(ld_train_x)
                )
            elif model_str == 'ld_shared':
                model = LD_SHARED(
                    h1_dim=args.hidden_dim,
                    num_labels=output,
                    num_neighbs=len(ld_train_x)
                )
            elif model_str == 'ld_indp':
                model = LD_INDP(
                    h1_dim=args.hidden_dim,
                    num_labels=output,
                    num_neighbs=len(ld_train_x)
                )
            else:
                raise Exception('model not implemented yet')

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            class_weights = torch.as_tensor([args.weighting_factor for _ in range(output)] if args.multilabel else [], dtype=torch.float)
            trainer = Trainer(
                batch_size=args.batch_size,
                device=device,
                log_interval=args.log_interval,
                multilabel=args.multilabel,
                class_weights=class_weights,
                patience=args.patience,
                optim_lr=args.lr,
                model=model
            )

            results_d = trainer.train(
                epochs=args.epochs,
                test_x=test_x,
                test_y=test_y,
                train_x=train_x,
                train_y=train_y,
                val_x=val_x,
                val_y=val_y)

            best_test_micro, best_test_macro = results_d['best_test_micro'], results_d[
                'best_test_macro']

            res[model_str].append([best_test_micro, best_test_macro])

    for m, scores in res.items():
        scores = np.array(scores)
        if args.dataset in multilabel_datasets:
            print(f'Model: {m}\n'
                  f'Micro F1: {np.mean(scores[:, 0])}+-{np.std(scores[:, 0])}, '
                  f'Macro F1: {np.mean(scores[:, 1])}+-{np.std(scores[:, 1])}')
        else:
            print(f'Model: {m}\n'
                  f'Accuracy: {np.mean(scores[:, 0])}+-{np.std(scores[:, 0])}')


