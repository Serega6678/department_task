from pathlib import Path
import sys

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
import torch
from torch.utils.data import DataLoader

from src.data import AuthorChangeFromNumpyDataset, collate_fn
from src.pl_module import LightningModule

if __name__ == "__main__":
    seed_everything(42)

    train_dataset_path = Path(sys.argv[1])
    train_dataset = AuthorChangeFromNumpyDataset(train_dataset_path)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=8)

    valid_dataset_path = Path(sys.argv[2])
    valid_dataset = AuthorChangeFromNumpyDataset(valid_dataset_path)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=8)

    pl_module = LightningModule()

    trainer = pl.Trainer(
        logger=WandbLogger(project="author_transition", save_dir="data"),
        callbacks=ModelCheckpoint(
            dirpath='data/checkpoints',
            filename='{epoch}-{val_loss:.2f}-{train_loss:.2f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True
        ),
        gpus=[0] if torch.cuda.is_available() else [],
        auto_lr_find=True,
        log_every_n_steps=3
    )
    trainer.fit(pl_module, train_loader, valid_loader)
