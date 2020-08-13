import pandas as pd
from pytorch_lightning.core.hooks import APEX_AVAILABLE
from sklearn.model_selection import train_test_split

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import config
from dataset import AptosDataset
from model import BaseCNNModel


class AptosModel(pl.LightningModule):

    def __init__(self, model):
        super(AptosModel, self).__init__()

        num_classes=5
        self.model = BaseCNNModel(num_classes)

        # define loss
        self.loss = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch    
        logits = self.model(x)
        loss = self.loss(logits, y)

        result = pl.TrainResult(loss)
        result.log('train_loss', loss)
        return result

    def prepare_data(self):
        fname = config.TRAIN_CSV_FILE
        df = pd.read_csv(fname)
        self.train, self.val = train_test_split(df,
                                                test_size=config.VAL_SIZE,
                                                random_state=42,
                                                shuffle=True,
                                                startify=df['diagnosis'])
        

    def train_dataloader(self):
        train_set = AptosDataset(self.train)
        train_loader = DataLoader(train_set,
                                  shuffle=True,
                                  num_workers=4,
                                  batch_size=config.BATCH_SIZE)

        return train_loader

    def val_dataloader(self):
        val_set = AptosDataset(self.val)
        val_loader = DataLoader(val_set,
                                num_workers=2,
                                batch_size=32)

        return val_loader

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        val_loss  = self.loss(logits, y)

        result = pl.EvalResult(checkpoint_on=val_loss)
        result.log('val_loss', val_loss)
        return result

    
if __name__ == "__main__":

    # create experiment
    experiment = AptosModel()

    # Create logger
    LOG = False
    if LOG:
        logger = WandbLogger(project="APTOS")
        logger.watch(experiment, log='all', log_freq=100)
    else:
        logger = None

    # create trainer
    trainer = pl.Trainer(max_epochs=2,
                         logger=logger,
                         progress_bar_refresh_rate=20)

    trainer.fit(experiment)