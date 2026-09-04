"""Lightning train/validate/test workflow using the restored PytorchModule."""
from pathlib import Path
import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from .PytorchModule import LitAutoEncoder


def seed_everything(seed=42):
    L.seed_everything(seed, workers=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_test(model, lr, epochs, train_loader, val_loader, test_loader,
               patience=30, version_name='interformer', dirpath='runs', *,
               accelerator='auto', enable_progress_bar=True):
    """Original positional calling convention; returns (trainer, logit_batches)."""
    if epochs < 1 or patience < 1:
        raise ValueError('epochs and patience must be positive.')
    run_dir = Path(dirpath) / version_name
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / 'checkpoints', filename='{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss', mode='min', save_top_k=1)
    trainer = L.Trainer(
        max_epochs=epochs, accelerator=accelerator, devices=1,
        gradient_clip_val=0.5, enable_progress_bar=enable_progress_bar,
        logger=CSVLogger(save_dir=str(run_dir), name='logs', version=0),
        callbacks=[checkpoint, EarlyStopping(monitor='val_loss', mode='min', patience=patience)])
    trainer.fit(LitAutoEncoder(model, lr), train_dataloaders=train_loader,
                val_dataloaders=val_loader)
    best = LitAutoEncoder.load_from_checkpoint(checkpoint.best_model_path, map_location='cpu')
    trainer.test(best, dataloaders=test_loader)
    outputs = trainer.predict(best, dataloaders=test_loader)
    return trainer, [output.detach().cpu() for output in outputs]


def load_model(path, device='cpu'):
    return LitAutoEncoder.load_from_checkpoint(path, map_location=device).model.to(device).eval()


@torch.no_grad()
def predict(model, loader, threshold=None):
    """Return probabilities and labels in loader order; use shuffle=False."""
    model.eval()
    device = next(model.parameters()).device
    probabilities = torch.cat([model(x.to(device)).softmax(-1).cpu() for x, _ in loader])
    if threshold is None:
        labels = probabilities.argmax(-1)
    else:
        if not 0 <= threshold <= 1:
            raise ValueError('threshold must be between 0 and 1.')
        labels = torch.where((probabilities[:, 0] > threshold) &
                             (probabilities[:, 0] > probabilities[:, 1]), 0, 1)
    return probabilities, labels
