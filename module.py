import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.metrics import Accuracy
import torchmetrics
import torch.nn.functional as F
from cifar10_models.densenet import densenet121, densenet161, densenet169
from cifar10_models.googlenet import googlenet
from cifar10_models.inception import inception_v3
from cifar10_models.mobilenetv2 import mobilenet_v2
from cifar10_models.resnet import resnet18, resnet34, resnet50
from cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from schduler import WarmupCosineLR

all_classifiers = {
    "vgg11_bn": vgg11_bn(),
    "vgg13_bn": vgg13_bn(),
    "vgg16_bn": vgg16_bn(),
    "vgg19_bn": vgg19_bn(),
    "resnet18": resnet18(num_classes=11, pretrained=True),
    "resnet34": resnet34(num_classes=11, pretrained=True),
    "resnet50": resnet50(num_classes=11, pretrained=True),
    "densenet121": densenet121(),
    "densenet161": densenet161(),
    "densenet169": densenet169(),
    "mobilenet_v2": mobilenet_v2(num_classes=11, pretrained=True),
    "googlenet": googlenet(),
    "inception_v3": inception_v3(),
}


def binary_focal(y_pred, y_true, weight=None, alpha=0.25, gamma=2):
    sigmoid_p = torch.sigmoid(y_pred)
    zeros = torch.zeros_like(sigmoid_p)
    pos_p_sub = torch.where(y_true > zeros, y_true - sigmoid_p, zeros)
    neg_p_sub = torch.where(y_true > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = -alpha * (pos_p_sub ** gamma) * torch.log(torch.clamp(sigmoid_p, 1e-8, 1.0)) - (1 - alpha) * (
            neg_p_sub ** gamma) * torch.log(torch.clamp(1.0 - sigmoid_p, 1e-8, 1.0))
    return per_entry_cross_ent.sum()


def _expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    bin_label_weights = label_weights.view(-1, 1).expand(
        label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights


class GHMC(nn.Module):
    def __init__(
            self,
            bins=10,
            momentum=0,
            use_sigmoid=True,
            loss_weight=1.0):
        super(GHMC, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = [float(x) / bins for x in range(bins + 1)]
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = [0.0 for _ in range(bins)]
        self.use_sigmoid = use_sigmoid
        self.loss_weight = loss_weight

    def forward(self, pred, target, label_weight, *args, **kwargs):
        """ Args:
        pred [batch_num, class_num]:
            The direct prediction of classification fc layer.
        target [batch_num, class_num]:
            Binary class target for each sample.
        label_weight [batch_num, class_num]:
            the value is 1 if the sample is valid and 0 if ignored.
        """
        if not self.use_sigmoid:
            raise NotImplementedError
        # the target should be binary class label
        if pred.dim() != target.dim():
            target, label_weight = _expand_binary_labels(target, label_weight, pred.size(-1))
        target, label_weight = target.float(), label_weight.float()
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(pred)

        # gradient length
        g = torch.abs(pred.sigmoid().detach() - target)

        valid = label_weight > 0
        tot = max(valid.float().sum().item(), 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                                      + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        loss = F.binary_cross_entropy_with_logits(
            pred, target, weights, reduction='sum') / tot
        return loss * self.loss_weight


class CIFAR10Module(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy()

        self.model = all_classifiers[self.hparams.classifier]

    def forward(self, batch):
        images, labels = batch
        predictions = self.model(images)
        loss = self.criterion(predictions, labels)
        accuracy = self.accuracy(predictions, labels)
        return loss, accuracy * 100

    def training_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("loss/train", loss)
        self.log("acc/train", accuracy)
        return loss

    def validation_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("loss/val", loss)
        self.log("acc/val", accuracy)

    def test_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("acc/test", accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        total_steps = self.hparams.max_epochs * len(self.train_dataloader())
        scheduler = {
            "scheduler": WarmupCosineLR(
                optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps
            ),
            "interval": "step",
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]


class MyModule(pl.LightningModule):
    def __init__(self, hparams, num_classes):
        super().__init__()
        # self.hparams = hparams
        self.save_hyperparameters(hparams)
        #         self.criterion = torch.nn.BCEWithLogitsLoss()
        #         self.criterion = binary_focal
        self.criterion = GHMC()
        self.accuracy = Accuracy(num_classes=num_classes, subset_accuracy=True, multiclass=True)
        self.train_acc = torchmetrics.Accuracy(num_classes=num_classes, subset_accuracy=True, multiclass=False)
        self.valid_acc = torchmetrics.Accuracy(num_classes=num_classes, subset_accuracy=True, multiclass=False)
        self.test_acc = torchmetrics.Accuracy(num_classes=num_classes, subset_accuracy=True, multiclass=False)
        self.model = all_classifiers[self.hparams.classifier]

    def forward(self, batch):
        images, labels = batch
        predictions = self.model(images)
        loss = self.criterion(predictions, labels)
        # accuracy = self.accuracy(predictions, torch.IntTensor(labels).cpu())
        return {'loss': loss, 'preds': predictions, 'target': labels}  # loss, accuracy * 100

    def training_step(self, batch, batch_nb):
        loss = self.forward(batch)['loss']  # , accuracy
        self.log("loss/train", loss)
        # self.log("acc/train", accuracy)
        return loss

    def training_step_end(self, outputs):
        # update and log
        preds = torch.IntTensor(outputs['preds']).cpu()
        train_acc = self.train_acc(preds, torch.IntTensor(outputs['target']).cpu())
        self.log('metric/train', train_acc)

    def validation_step(self, batch, batch_nb):
        loss = self.forward(batch)['loss']  # , accuracy
        self.log("loss/val", loss)
        # self.log("acc/val", accuracy)

    def validation_step_end(self, outputs):
        # update and log
        preds = torch.IntTensor(outputs['preds']).cpu()
        valid_acc = self.valid_acc(preds, torch.IntTensor(outputs['target']).cpu())
        self.log('metric/val', valid_acc)

    def test_step(self, batch, batch_nb):
        loss = self.forward(batch)['loss']  # , accuracy
        # self.log("acc/test", accuracy)

    def test_step_end(self, outputs):
        # update and log
        preds = torch.IntTensor(outputs['preds']).cpu()
        test_acc = self.test_acc(preds, torch.IntTensor(outputs['target']).cpu())
        self.log('metric/test', test_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        total_steps = self.hparams.max_epochs * len(self.train_dataloader())
        scheduler = {
            "scheduler": WarmupCosineLR(
                optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps
            ),
            "interval": "step",
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]
