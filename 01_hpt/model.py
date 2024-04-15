
import torch
import timm

import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn

from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix

def get_model(model_name):  
    if model_name == "resnet18":
        model = timm.create_model("resnet18", pretrained=True, num_classes=6)
    elif model_name == "resnet34":
        model = timm.create_model("resnet34", pretrained=True, num_classes=6)
    elif model_name == "resnet26":
        model = timm.create_model("resnet26", pretrained=True, num_classes=6)
    return model

class LitResnet(pl.LightningModule):
    def __init__(self, model, lr, opt, num_classes=6):
        super().__init__()

        self.save_hyperparameters()
        self.num_classes = num_classes
        self.model = get_model(self.hparams.model)
        self.loss = nn.CrossEntropyLoss()
        self.train_acc = MulticlassAccuracy(num_classes=6)
        self.val_acc = MulticlassAccuracy(num_classes=6)
        self.test_acc = MulticlassAccuracy(num_classes=6)
        self.conf_mat = MulticlassConfusionMatrix(num_classes=6)

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        train_loss = self.loss(logits, y)
        train_acc = self.train_acc(logits, y)
        # self.log('train_acc_step', train_acc)
        # self.log('train_loss_step', train_loss)
        return {"loss": train_loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        val_loss = self.loss(logits, y)
        val_acc = self.val_acc(logits, y)
        # self.log('val_acc_step', val_acc)
        # self.log('val_loss_step', val_loss)
        return {"loss": val_loss}
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        test_loss = self.loss(logits, y)
        test_acc = self.test_acc(preds, y)
        # self.log('test_acc_step', test_acc)
        # self.log('test_loss_step', test_loss)
        return {"loss": test_loss, "test_preds": preds, "test_targ": y}
    
    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.hstack([x['loss'] for x in outputs]).mean()
        self.log('valid_loss_epoch', avg_val_loss)
        self.log('valid_acc_epoch', self.val_acc.compute())
        self.logger.experiment.add_hparams({'model': {self.hparams.model}, 'opt': {self.hparams.opt}, 'lr': {self.hparams.lr}},
                                           {'accuracy': self.val_acc.compute(), 'loss': avg_val_loss})
        self.val_acc.reset()
        
    def test_epoch_end(self, outputs):
        avg_test_loss = torch.hstack([x['loss'] for x in outputs]).mean()
        self.log('test_loss_epoch', avg_test_loss)
        self.log('test_acc_epoch', self.test_acc.compute())
        self.test_acc.reset()
        # preds = torch.cat([x['test_preds'] for x in outputs])
        # targs = torch.cat([x['test_targ'] for x in outputs])       
        # confmat = self.conf_mat(preds, targs)
        # torch.save(confmat, f"test-confmat.pt")
        
    def training_epoch_end(self, outputs):
        avg_train_loss = torch.hstack([x['loss'] for x in outputs]).mean()        
        self.log('train_loss_epoch', avg_train_loss)
        self.log('train_acc_epoch', self.train_acc.compute())
        
        self.train_acc.reset()

    def configure_optimizers(self):
        
        if self.hparams.opt == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.lr,
                momentum=0.9,
                weight_decay=5e-4,
            )
        elif self.hparams.opt == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.lr,
            )
        
        return {"optimizer": optimizer}