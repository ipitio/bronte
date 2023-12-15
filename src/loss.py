import torch
from torch import nn


class DeepLoss(nn.Module):
    def __init__(self, criterion, num_out=[1]):
        super(DeepLoss, self).__init__()
        self.criterion = criterion
        self.num_out = num_out

    def forward(self, preds, y):
        if not isinstance(preds, list):
            preds = [
                preds
            ]  # For single-input, shap doesn't expect a list; the following does

        losses = []
        for i, pred in enumerate(preds):
            # if multi-output, then the task is classification
            if self.num_out[i] > 1:
                _, pred = torch.max(pred, 1)
            target = y[:, i]
            while target.ndim > pred.ndim:
                target = target.squeeze(1)
            while target.ndim < pred.ndim:
                target = target.unsqueeze(1)
            try:
                losses.append(self.criterion(pred.float(), target[: len(pred)].float()))
            except:
                losses.append(self.criterion(pred.float(), target[: len(pred)].long()))

        return (
            losses[0] if len(losses) < 2 else torch.stack(losses).mean(dim=0)
        ).mean()
