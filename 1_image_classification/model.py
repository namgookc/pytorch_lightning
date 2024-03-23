import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl

model_dict = {}
act_fn_by_name = {
    'tanh': nn.Tanh,
    'relu': nn.ReLU,
    'leakyrelu': nn.LeakyReLU,
    'gelu': nn.GELU
}


class CIFARModule(pl.LightningModule):
    def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams):
        '''
        Inputs:
            model_name - Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary which includes learning rate, weight decay, etc.
        '''
        super().__init__()

        # exports the hyperparameters to a YAML file, and create 'self.hparams' namespace
        self.save_hyperparameters()

        # create model: mapping the model-name to the model-class
        self.model = create_model(model_name, model_hparams)

        # create loss module
        self.loss_module = nn.CrossEntropyLoss()

        # example input for visualizing the graph in tensorboard
        self.example_input_array = torch.zeros((1,3,32,32), dtype=torch.float32)

    def forward(self, imgs):
        return self.model(imgs)

    def configure_optimizers(self):
        if self.hparams.optimizer_name == 'Adam':
            # AdamW is Adam with a correct implementation of weight decay
            # (see here for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == 'SGD':
            optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        else:
            assert False, f'Unknown optimizer: \'{self.hparams.optimizer_name}\''

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)
        #return [optimizer], [scheduler]
        return optimizer

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1)==labels).float().mean()

        # log the accuracy per epoch to tensorboard (weighted average over batches)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss)
        return loss  # return tensor to call '.backward' on

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (preds==labels).float().mean()
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (preds==labels).float().mean()
        self.log('test_acc', acc)


def create_model(model_name, model_hparams):
    if model_name in model_dict:
        return model_dict[model_name](**model_hparams) # it maps the model-name to the model-class
    else:
        assert False, f'Unknown model name \'{model_name}\'. Available models are: {str(model_dict.keys())}'
