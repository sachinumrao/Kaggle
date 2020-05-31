import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import BertModel
from transformers import BertTokenizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


# Dataset class
class BertDataset:
    def __init__(self, text, target, tokenizer, max_len):
        self.text = text
        self.target = target
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text)

        inputs = self.tokenizer.encode_plus(text, None, add_special_tokens=True, max_length=self.max_len)
        ids = inputs["input_ids"]
        masks = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        pad_length = self.max_len - len(ids)
        ids = ids + ([0] * pad_length)
        masks = masks + ([0] * pad_length)
        token_type_ids = token_type_ids + ([0] * pad_length)

        data = {
            'ids': torch.tensor(ids, dtype=torch.long),
            'masks': torch.tensor(masks, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.target[item], dtype=torch.float)
        }
        return data


# Model class
class BertModelClassifier(pl.LightningModule):

    def __init__(self, config):
        super(BertModelClassifier, self).__init__()

        self.config = config
        self.train_dataset = None
        self.val_dataset = None

        # Define Bert Model
        self.bert_layer = BertModel.from_pretrained(config['BERT_MODEL'])
        # Freeze the bert model
        for param in self.bert_layer.parameters():
            param.requires_grad = False

        # Define tokenizer
        self.bert_tokenizer = BertTokenizer.from_pretrained(config['BERT_MODEL'], do_lower_case=True)

        # Add dense layers on top of bert model
        self.linear_layer1 = nn.Linear(768, 64)
        self.linear_layer2 = nn.Linear(64, 1)

        # Define loss
        self.loss_layer = nn.BCEWithLogitsLoss()

    def forward(self, ids, masks, token_type_ids):
        _, h = self.bert_layer(ids, attention_mask=masks, token_type_ids=token_type_ids)

        h = self.linear_layer1(h)
        out = self.linear_layer2(h)
        return out

    def accuracy_fn(self, logits, y):
        threshold = 0.5
        y = y.numpy().reshape((-1, ))
        y_pred = logits.detach().numpy().reshape((-1, ))
        y_pred = np.where(y_pred > threshold, 1.0, 0.0)
        acc = accuracy_score(y, y_pred)
        acc = torch.tensor(acc, dtype=torch.float32)
        return acc

    def configure_optimizers(self):
        lr = self.config.get('LR', 0.001)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer

    def loss_fn(self, logits, y):
        y = y.view(-1, 1)
        loss = self.loss_layer(logits, y)
        return loss

    @pl.data_loader
    def prepare_data(self):
        val_size = self.config.get('VAL_SIZE', 0.10)
        # load data from csv file
        dfx = pd.read_csv(self.config['TRAIN_FILE']).fillna("none")
        df_train, df_valid = train_test_split(dfx,
                                              test_size=self.config['VAL_SIZE'],
                                              random_state=self.config['SEED'],
                                              stratify=dfx.target.values)

        df_train = df_train.reset_index(drop=True)
        df_valid = df_valid.reset_index(drop=True)

        self.train_dataset = BertDataset(text=df_train['clean_text'].values,
                                         target=df_train['target'].values,
                                         tokenizer=self.bert_tokenizer,
                                         max_len=self.config['MAX_LEN'])

        self.val_dataset = BertDataset(text=df_valid['clean_text'].values,
                                       target=df_valid['target'].values,
                                       tokenizer=self.bert_tokenizer,
                                       max_len=self.config['MAX_LEN'])


    @pl.data_loader
    def train_dataloader(self):
        dataloader = DataLoader(self.train_dataset,
                                batch_size=self.config['TRAIN_BATCH_SIZE'],
                                shuffle=True,
                                num_workers=8)
        return dataloader


    @pl.data_loader
    def val_dataloader(self):
        dataloader = DataLoader(self.val_dataset,
                                batch_size=self.config['VALID_BATCH_SIZE'],
                                shuffle=False,
                                num_workers=4)
        return dataloader

    def training_step(self, batch, batch_idx):
        # Run forward pass
        out = self.forward(batch['ids'],
                           batch['masks'],
                           batch['token_type_ids'])
        y = batch['targets']

        # Calculate loss
        loss = self.loss_fn(out, y)

        # Calculate accuracy
        acc = self.accuracy_fn(out, y)

        # Logs
        logs = {
            "loss": loss,
            "acc": acc,
        }
        results = {
            "loss": loss,
            "acc": acc,
            "log": logs
        }
        return results

    def train_epoch_end(self, outputs):
        loss = torch.stack([x['train_loss'] for x in outputs]).mean()
        acc = torch.stack([x['train_acc'] for x in outputs]).mean()
        logs = {'train_loss': loss,
                'train_acc': acc}
        results = {'train_loss': loss, 'train_acc': acc, 'log': logs}
        return results

    def validation_step(self, batch, batch_idx):
        # Run forward pass
        out = self.forward(batch['ids'],
                           batch['masks'],
                           batch['token_type_ids'])
        y = batch['targets']

        # Calculate loss
        val_loss = self.loss_fn(out, y)

        # Calculate accuracy
        val_acc = self.accuracy_fn(out, y)

        # Logs
        logs = {
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        results = {
            "val_loss": val_loss,
            "val_acc": val_acc,
            "log": logs
        }
        return results

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss,
                'val_acc': avg_acc}
        results = {'val_loss': avg_loss, 'val_acc': avg_acc, 'log': logs}
        return results


if __name__ == "__main__":
    # Define configurations of Bert model
    model_config = {
        "MAX_LEN": 512,
        "TRAIN_BATCH_SIZE": 8,
        "VALID_BATCH_SIZE": 4,
        "EPOCHS": 5,
        "LR": 0.0005,
        "VAL_SIZE": 0.10,
        "BERT_MODEL": 'bert-base-uncased',
        "MODEL_PATH": '~/Data/Kaggle/real_or_not/',
        "TRAIN_FILE": '~/Data/Kaggle/real_or_not/clean_train.csv',
        "SEED": 99
    }

    pl.seed_everything(model_config['SEED'])

    # Create Model
    model = BertModelClassifier(model_config)
    model.prepare_data()

    # Create logger
    wandb_logger = WandbLogger(project="RealTweets")
    wandb_logger.watch(model, log='all', log_freq=100)

    # Create model trainer
    trainer = pl.Trainer(max_epochs=10, logger=wandb_logger, deterministic=True, profiler=True)

    # Train the model
    trainer.fit(model)

