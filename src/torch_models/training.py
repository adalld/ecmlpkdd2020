import torch
from src.utils import make_sure_path_exists
import os
import time
from sklearn.metrics import f1_score, roc_auc_score
from src.Variables import BINARY_CLASSIFICATION_PROBABILITY_THRESHOLD

VAL_ACC_KEY = 'early_stopping_best_val_accuracy'
class Trainer:
    def __init__(self,
                 batch_size: int,
                 device: torch.device,
                 log_interval: int,
                 multilabel: bool,
                 patience: int,
                 optim_lr: float,
                 model: torch.nn.Module,
                 init_model_new_for_each_train_run:bool = False,
                 class_weights: torch.Tensor = None,
                 checkpoint_path: str = '../checkpoints/'
                 ):
        self.batch_size = batch_size
        self.device = device
        self.log_interval = log_interval
        if not multilabel:
            assert class_weights.nelement() == 0 # only for multilabel
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights).to(device) if multilabel else torch.nn.CrossEntropyLoss()
        self.patience = patience
        make_sure_path_exists(checkpoint_path)
        ts = int(time.time() * 1000.0)
        self.checkpoint_path = os.path.join(checkpoint_path, f'ckp_{ts}')
        self.model = model.to(device)
        self.multilabel = multilabel
        self.best_scores_among_runs = {VAL_ACC_KEY: float('-inf')}
        self.lr = optim_lr
        self.init_model_new_for_each_train_run = init_model_new_for_each_train_run
        self.reset_state()

    def reset_state(self):
        self.model.reset_parameters()
        self.optim = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)



    def train(self,
              epochs: int,
              test_x: torch.FloatTensor,
              test_y: torch.LongTensor,
              train_x: torch.LongTensor,
              train_y: torch.LongTensor,
              val_x: torch.FloatTensor,
              val_y: torch.LongTensor):
        if self.init_model_new_for_each_train_run:
            self.reset_state()
        early_stopping_counter = 0
        best_early_stopping_score =  float('-inf')
        for e in range(epochs):
            perm = torch.randperm(train_x.shape[1])
            train_x = train_x[:, perm, :]
            train_y = train_y[perm]
            train_losses = []
            self.model.train()
            for start_ind in range(0, train_x.shape[1], self.batch_size):
                end_ind = min(train_x.shape[1], start_ind + self.batch_size)
                current_x = train_x[:, start_ind:end_ind, :]
                current_y = train_y[start_ind:end_ind]
                self.optim.zero_grad()
                outputs = self.model(ld_input=current_x.to(self.device))
                train_loss = self.loss_fn(input=outputs, target=current_y.to(self.device))
                train_losses.append(train_loss.item())
                train_loss.backward()
                self.optim.step()

            if epochs > 0:
                train_loss, train_micro, train_macro = self._eval(x=train_x, y=train_y)
                val_loss, val_micro, val_macro = self._eval(x=val_x, y=val_y)
                test_loss, test_micro, test_macro = self._eval(x=test_x, y=test_y)
                if e > 0 and e % self.log_interval == 0:
                    print(f'epoch {e}:'
                          f'\ntrain: loss {train_loss}, microf1 {train_micro}, macro {train_macro}'
                          f'\nval: loss {val_loss}, microf1 {val_micro}, macro {val_macro}'
                          f'\ntest: loss {test_loss}, microf1 {test_micro}, macro {test_macro}')
                if val_micro > best_early_stopping_score:
                    early_stopping_counter = 0
                    best_early_stopping_score = val_micro
                    if best_early_stopping_score > self.best_scores_among_runs[VAL_ACC_KEY]:
                        self.best_scores_among_runs[VAL_ACC_KEY] = best_early_stopping_score
                        self.best_scores_among_runs['best_test_micro'] = test_micro
                        self.best_scores_among_runs['best_test_macro'] = test_macro
                        self.save_state()
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= self.patience:
                        print(f'Early stopping after {e} epochs....')
                        break
        self.load_state()
        return self.best_scores_among_runs.copy()

    def _eval(self,
              x: torch.FloatTensor,
              y: torch.LongTensor,
             ):
        self.model.eval()
        x = x.to(self.device)
        y = y.to(self.device)
        outputs = self.model(x)
        loss = self.loss_fn(input=outputs, target=y)
        y = y.cpu().detach()
        if self.multilabel:
            pred = outputs.cpu().detach() >= BINARY_CLASSIFICATION_PROBABILITY_THRESHOLD
        else:
            pred = torch.argmax(outputs, dim=1).cpu().detach()
        micro, macro = f1_score(y_true = y, y_pred=pred, average='micro'), f1_score(y_true = y, y_pred=pred, average='macro')
        #else:
        #accuracy = float(torch.sum(torch.argmax(outputs, dim=1) == y)) / float(x.shape[1])
        return loss, micro, macro

    def save_state(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
        }, self.checkpoint_path)

    def load_state(self):
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])

    #TODO merge both methods
    def predict(self,
                x: torch.FloatTensor,
                y: torch.LongTensor):
        x = x.to(self.device)
        y = y.to(self.device)
        self.model.eval()
        outputs = self.model(x)
        if self.multilabel:
            preds = outputs.cpu().detach() >= BINARY_CLASSIFICATION_PROBABILITY_THRESHOLD
        else:
            preds = torch.argmax(outputs, dim=1).cpu().detach()
        micro, macro = f1_score(y_true=y.cpu().detach(), y_pred=preds, average='micro'), \
                       f1_score(y_true=y.cpu().detach(), y_pred=preds, average='macro')

        return preds, micro, macro

    def get_predictions(self,
                        x: torch.FloatTensor,
                        y: torch.LongTensor):
        x = x.to(self.device)
        y = y.to(self.device)
        self.model.eval()
        outputs = self.model(x)
        preds = outputs.cpu().detach() >= BINARY_CLASSIFICATION_PROBABILITY_THRESHOLD
        micro, macro = f1_score(y_true=y, y_pred=preds, average='micro'), f1_score(y_true=y, y_pred=preds, average='macro')

        return outputs.cpu().detach(), micro, macro

    def max_prediction(self,
                       x: torch.FloatTensor):
        self.model.eval()
        x = x.to(self.device)
        outputs = self.model(x)
        return torch.max(torch.nn.functional.softmax(outputs, dim=1), dim=1)

