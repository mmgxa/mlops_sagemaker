import io

import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torchmetrics.classification import (MulticlassAccuracy, 
                                        MulticlassConfusionMatrix, MulticlassPrecision,
                                        MulticlassRecall, MulticlassF1Score)


import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from PIL import Image

import timm
from mlxtend.plotting import plot_confusion_matrix

def get_model(model_name):  
    if model_name == "resnet18":
        model = timm.create_model("resnet18", pretrained=True, num_classes=6)
    elif model_name == "resnet34":
        model = timm.create_model("resnet34", pretrained=True, num_classes=6)
    elif model_name == "resnet50":
        model = timm.create_model("resnet50", pretrained=True, num_classes=6)
    return model

def create_cm_img(conf_mat):
    class_list = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']    
    fig, ax = plot_confusion_matrix(conf_mat=conf_mat.numpy(),
                                colorbar=False,
                                show_absolute=True,
                                show_normed=False,
                                class_names=class_list);
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg', bbox_inches='tight')
    buf.seek(0)
    im = Image.open(buf)
    im = torchvision.transforms.ToTensor()(im)
    return im


class LitResnet(pl.LightningModule):
    def __init__(self, model, lr, opt, num_classes=6):
        super().__init__()

        self.save_hyperparameters()
        self.num_classes = num_classes
        # self.model = timm.create_model('resnet18', pretrained=True, num_classes=num_classes)
        self.model = get_model(self.hparams.model)
        self.loss = nn.CrossEntropyLoss()
        
        self.train_acc = MulticlassAccuracy(num_classes=6)
        self.val_acc = MulticlassAccuracy(num_classes=6)
        self.test_acc = MulticlassAccuracy(num_classes=6)
        
        self.train_pr = MulticlassPrecision(num_classes=6)
        self.val_pr = MulticlassPrecision(num_classes=6)
        self.test_pr = MulticlassPrecision(num_classes=6)
        
        self.train_rec = MulticlassRecall(num_classes=6)
        self.val_rec = MulticlassRecall(num_classes=6)
        self.test_rec = MulticlassRecall(num_classes=6)
        
        self.train_f1 = MulticlassF1Score(num_classes=6)
        self.val_f1 = MulticlassF1Score(num_classes=6)
        self.test_f1 = MulticlassF1Score(num_classes=6)
        
        self.train_conf_mat = MulticlassConfusionMatrix(num_classes=6)
        self.val_conf_mat = MulticlassConfusionMatrix(num_classes=6)
        self.test_conf_mat = MulticlassConfusionMatrix(num_classes=6)

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        train_loss = self.loss(logits, y)
        train_acc = self.train_acc(preds, y)
        train_pr = self.train_pr(preds, y)
        train_rec = self.train_rec(preds, y)
        train_f1 = self.train_f1(preds, y)
        train_conf_mat = self.train_conf_mat(preds, y)
        # self.log('train_acc_step', train_acc)
        # self.log('train_loss_step', train_loss)
        return {"loss": train_loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        val_loss = self.loss(logits, y)
        val_acc = self.val_acc(preds, y)
        val_pr = self.val_pr(preds, y)
        val_rec = self.val_rec(preds, y)
        val_f1 = self.val_f1(preds, y)
        val_conf_mat = self.val_conf_mat(preds, y)
        # self.log('val_acc_step', val_acc)
        # self.log('val_loss_step', val_loss)
        return {"loss": val_loss}
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        test_loss = self.loss(logits, y)
        test_acc = self.test_acc(preds, y)
        test_pr = self.test_pr(preds, y)
        test_rec = self.test_rec(preds, y)
        test_f1 = self.test_f1(preds, y)
        test_conf_mat = self.test_conf_mat(preds, y)
        # self.log('test_acc_step', test_acc)
        # self.log('test_loss_step', test_loss)
        return {"loss": test_loss, "test_preds": preds, "test_targ": y}
    
    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.hstack([x['loss'] for x in outputs]).mean()
        self.log('valid/loss_epoch', avg_val_loss, sync_dist=True)
        self.log('valid/acc_epoch', self.val_acc.compute(), sync_dist=True)
        self.val_acc.reset()
        self.log('valid/pr_epoch', self.val_pr.compute(), sync_dist=True)
        self.log('valid/rec_epoch', self.val_rec.compute(), sync_dist=True)
        self.log('valid/f1_epoch', self.val_f1.compute(), sync_dist=True)
        self.val_pr.reset()
        self.val_rec.reset()
        self.val_f1.reset()
        confmat = self.val_conf_mat.compute().cpu()
        im = create_cm_img(confmat)
        self.logger.experiment.add_image(f"valid/confusion_matrix_{self.current_epoch}",
                                        im,
                                        global_step=self.current_epoch,
                                        )
        self.val_conf_mat.reset()
        
    def test_epoch_end(self, outputs):
        avg_test_loss = torch.hstack([x['loss'] for x in outputs]).mean()
        self.log('test/loss_epoch', avg_test_loss, sync_dist=True)
        self.log('test/acc_epoch', self.test_acc.compute(), sync_dist=True)
        self.test_acc.reset()
        self.log('test/pr_epoch', self.test_pr.compute(), sync_dist=True)
        self.log('test/rec_epoch', self.test_rec.compute(), sync_dist=True)
        self.log('test/f1_epoch', self.test_f1.compute(), sync_dist=True)
        self.test_pr.reset()
        self.test_rec.reset()
        self.test_f1.reset()
        confmat = self.test_conf_mat.compute().cpu()
        im = create_cm_img(confmat)
        self.logger.experiment.add_image(f"test/confusion_matrix",
                                        im,
                                        global_step=self.current_epoch
                                        )
        self.test_conf_mat.reset()
        
    def training_epoch_end(self, outputs):
        avg_train_loss = torch.hstack([x['loss'] for x in outputs]).mean()
        self.log('train/loss_epoch', avg_train_loss, sync_dist=True)
        self.log('train/acc_epoch', self.train_acc.compute(), sync_dist=True)
        self.train_acc.reset()
        self.log('train/pr_epoch', self.train_pr.compute(), sync_dist=True)
        self.log('train/rec_epoch', self.train_rec.compute(), sync_dist=True)
        self.log('train/f1_epoch', self.train_f1.compute(), sync_dist=True)
        self.train_pr.reset()
        self.train_rec.reset()
        self.train_f1.reset()
        confmat = self.train_conf_mat.compute().cpu()
        im = create_cm_img(confmat)
        self.logger.experiment.add_image(f"train/confusion_matrix_{self.current_epoch}",
                                        im,
                                        global_step=self.current_epoch,
                                        )
        self.train_conf_mat.reset()

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
