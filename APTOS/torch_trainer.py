import torch
import torch.optim as optim 
import torch.nn as nn   
import torch.nn.functional as F  
from torch.utils.data import DataLoader

import time
from tqdm import tqdm
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

import config
from dataset import AptosDataset
from model import BaseCNNModel


class Trainer(object):

    def __init__(self, model, data_file, lr=1e-3, max_epochs=5, batch_size=32, log_steps=10, n_folds=5, cv=False):

        # instantiate model and other training attributes
        self.data_file = data_file
        self.model = model
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.log_steps = log_steps
        self.cv = cv
        self.n_folds = n_folds
        self.cv_data = None
        self.df = None
        self.train = None
        self.valid = None

        # loss
        self.loss_layer = nn.CrossEntropyLoss()

        # performance logging
        self.train_loss = []
        self.val_loss = []

    def _load_data(self):
        # load data
        df = pd.read_csv(self.data_file, nrows=1000)

        if not self.cv:
            # if cross validation is not ture then do simple train test split
            self.train, self.valid = train_test_split(df,
                                                      test_size=config.VAL_SIZE,
                                                      random_state=42,
                                                      shuffle=True,
                                                      stratify=df['diagnosis'])

        else:
            self.df = df.sample(frac=1.0)
            self.cv_data = df
            
    def _get_fold_data(self, idx):
        val_len = self.df.shape[0] // self.n_folds
        valid = self.df.iloc[idx*val_len : (idx+1)*val_len]
        train = self.df.drop(valid.index)
        return (train, valid)
            

    def _set_train_loader(self, batch_size=32):
        train_set = AptosDataset(self.train)
        self.train_loader = DataLoader(train_set,
                                  shuffle=True,
                                  num_workers=4,
                                  batch_size=batch_size)


    def _set_val_loader(self):
        val_set = AptosDataset(self.valid)
        self.val_loader = DataLoader(val_set,
                                num_workers=2,
                                batch_size=32)

    def _set_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def simple_fit(self):
        self._load_data()
        self._set_train_loader(self.batch_size)
        self._set_val_loader()
        self._set_optimizer()
        # reset model
        self._reset_model()
        results = self._run_experiment()
        return results

    def _run_experiment(self):
        for epoch in range(self.max_epochs):

            # run train loop
            t_epoch_begin = time.time()
            print("Epoch: ", epoch+1)
            self.model.train()
            losses = []

            for step, batch in enumerate(self.train_loader):
                x, y = batch
                # step-1: forward pass
                logits = self.model(x)
                # step-2: measure loss
                loss = self.loss_layer(logits, y)
                # step-3: clear gradients
                self.model.zero_grad()
                # step-4: calculate gradients
                grads = loss.backward()
                # step-5: apply grads by backprop
                self.optimizer.step(grads)

                # log train loss
                losses.append(loss.item())

                # print train logs
                if (step+1)%self.log_steps == 0:
                    self.train_loss.append(torch.tensor(losses).mean().detach().numpy())
                    print(f"Step: {step+1} Train_Loss: {self.train_loss[-1]:.3f}")

            t_epoch_end = time.time()
            print(f"Time Taken: {(t_epoch_end - t_epoch_begin):.1f} sec")

            # run validation loop
            print("Running Validation Step...")
            self.model.eval()
            val_losses = []
            for batch in self.val_loader:
                x, y = batch
                with torch.no_grad():
                    logits = self.model(x)
                loss = self.loss_layer(logits, y)
                val_losses.append(loss.item())
            
            self.val_loss.append(torch.tensor(val_losses).mean().detach().numpy())
            print(f"Epoch {epoch+1} Validation Loss: {self.val_loss[-1]:.3f}")
            print()

        results = {
                    'train_loss': self.train_loss,
                    'val_loss': self.val_loss
                  }

        return results

    def _reset_model(self):
        for name, module in model.named_children():
            try:
                # print('resetting ', name)
                module.reset_parameters()
            except:
                continue

    def cv_fit(self):
        cv_results = []
        cv_perf = {}
        # load data into the class
        self._load_data()

        for i in range(self.n_folds):
            print("Fold: ", i+1)
            train, valid = self._get_fold_data(i)
            self.train = train
            self.valid = valid
            self._set_train_loader(self.batch_size)
            self._set_val_loader()
            self._set_optimizer()
            # reset model
            self._reset_model()
            exp_results = self._run_experiment()
            cv_results.append(exp_results)

        for i in range(self.n_folds):
            res = cv_results[i]
            avg_val_loss = np.mean(res['val_loss'][-2:])
            cv_perf[i] = avg_val_loss

        cv_perf_df = pd.DataFrame(cv_perf.items(), columns=['Fold', 'Val_Loss'])

        return cv_perf_df, cv_results

    @staticmethod
    def plot_training_performance(results):
        plt.plot(results['train_loss'])
        plt.show()

        plt.plot(results['val_loss'])
        plt.show()



if __name__ == "__main__":

    num_classes = config.NUM_CLASSES
    model = BaseCNNModel(num_classes)

    lr = config.LR
    num_epochs = config.NUM_EPOCHS
    log_steps = config.LOG_STEPS
    is_cv = config.isCV
    n_folds = config.N_FOLDS

    datafile = config.TRAIN_CSV_FILE

    trainer = Trainer(model, 
                      datafile,
                      lr=lr,
                      max_epochs=num_epochs,
                      log_steps=log_steps,
                      n_folds=n_folds, 
                      cv=is_cv)

    # cv_perf_df, cv_results = trainer.cv_fit()
    # print(cv_perf_df)
    # print(cv_results)
    results = trainer.simple_fit()
    trainer.plot_training_performance(results)



