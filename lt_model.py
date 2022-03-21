import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from dpt.loss import ScaleAndShiftInvariantLoss
from dpt.models import DPTDepthModel


class DPTDepthWrapper(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.model = DPTDepthModel(
            path=None,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        self.loss_fn = ScaleAndShiftInvariantLoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.loss_fn(y_hat, y)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = self.loss_fn(y_hat, y)
        self.log("test_loss", test_loss)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        return pred
