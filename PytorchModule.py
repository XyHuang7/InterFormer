"""Lightning wrapper adapted from the original CV_Transformer/PytorchModule.py."""
import lightning as L
import torch
from torchmetrics.classification import BinaryAccuracy
from .model import InterFormer


class LitAutoEncoder(L.LightningModule):

    def __init__(self, model=None, lr=1e-3, l1_lambda=0., model_config=None):
        super().__init__()
        if model is None:
            if model_config is None:
                raise ValueError('Provide model or model_config.')
            model = InterFormer(**model_config)
        self.model = model
        self.lr, self.l1_lambda = lr, l1_lambda
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()
        self.loss = torch.nn.CrossEntropyLoss()
        self.save_hyperparameters(dict(model_config=model.config, lr=lr, l1_lambda=l1_lambda))
        self.gradient_norm_sum = 0.
        self.num_steps = 0

    def forward(self, inputs):
        x = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
        return self.model(x)

    def _step(self, batch, stage):
        x, y = batch
        labels = y.argmax(-1) if y.ndim == 2 else y.long()
        logits = self(x)
        loss = self.loss(logits, labels)
        if stage == 'train' and self.l1_lambda:
            loss = loss + self.l1_lambda * sum(p.abs().sum() for p in self.model.parameters())
        metric = getattr(self, f'{stage}_acc')
        metric.update(logits.argmax(-1), labels)
        self.log(f'{stage}_acc', metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f'{stage}_loss', loss, on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=len(labels))
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self._step(batch, 'test')

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def on_train_epoch_start(self):
        self.gradient_norm_sum = 0.
        self.num_steps = 0

    def on_before_optimizer_step(self, optimizer):
        norms = [p.grad.detach().norm(2) for p in self.model.parameters() if p.grad is not None]
        if norms:
            norm = torch.stack(norms).norm(2)
            self.log('gradient', norm, on_step=True, on_epoch=False)
            self.gradient_norm_sum += norm.item()
            self.num_steps += 1

    def on_train_epoch_end(self):
        if self.num_steps:
            self.log('avg_gradient_norm', self.gradient_norm_sum / self.num_steps)

    def configure_optimizers(self):
        # Original scheduler was never returned; retain the actually active Adam.
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
