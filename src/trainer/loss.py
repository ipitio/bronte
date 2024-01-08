import torch


class DeepLoss:
    def __init__(self, criterion, num_out=[1]):
        self.criterion = criterion
        self.num_out = num_out

    def mean(self, preds, y):
        if not isinstance(preds, list):
            preds = [
                preds
            ]  # For single-input, shap doesn't expect a list; the following does

        losses = []
        for i, pred in enumerate(preds):
            # if multi-output, then the task is classification
            if self.num_out[i] > 1:
                pred = pred.argmax(dim=1)
            target = y[:, i]
            while target.ndim > pred.ndim:
                target = target.squeeze(1)
            while target.ndim < pred.ndim:
                target = target.unsqueeze(1)
            minlen = min(len(pred), len(target))
            pred = pred[:minlen]
            target = target[:minlen]

            try:
                losses.append(self.criterion(pred.float(), target.float()))
            except:
                losses.append(self.criterion(pred.float(), target.long()))

        return (
            losses[0] if len(losses) < 2 else torch.stack(losses).mean(dim=0)
        ).mean()
