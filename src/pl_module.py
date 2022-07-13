import typing as tp

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from src.model import AuthorClassifier


class LightningModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.model = AuthorClassifier(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: tp.Any) -> torch.Tensor:
        text_embeddings = batch["embeddings"]
        labels = batch["labels"]
        logits = self.forward(text_embeddings)
        logits = logits[labels != -1]
        labels = labels[labels != -1]
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: tp.Any) -> torch.Tensor:
        text_embeddings = batch["embeddings"]
        labels = batch["labels"]
        logits = self.forward(text_embeddings)
        logits = logits[labels != -1]
        labels = labels[labels != -1]
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self) -> tp.Dict[
        str, tp.Union[torch.optim.Optimizer, torch.optim.lr_scheduler.ConstantLR]
    ]:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    factor=0.1,
                    patience=3,
                    verbose=True,
                    cooldown=5,
                ),
                "frequency": 1,
                "monitor": "val_loss",
                "interval": "epoch"
            },
        }
