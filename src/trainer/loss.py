import torch


class DeepLoss:
    def __init__(self, criterion, num_out=[1]):
        self.criterion = criterion
        self.num_out = num_out

    def mean(self, preds, y):
        losses = []
        if not isinstance(preds, list):
            preds = [
                preds
            ]  # For single-input, shap doesn't expect a list; the following does

        for i, pred in enumerate(preds):
            target = y[:, i]
            minlen = min(len(pred), len(target))
            pred = pred[:minlen]
            target = target[:minlen]

            try:
                losses.append(self.criterion(pred.float(), target.float()))
            except:
                try:
                    losses.append(self.criterion(pred.float(), target.long()))
                except:
                    target = target.to(pred.dtype)
                    losses.append(self.criterion(pred.float(), target))

        return (losses[0] if len(losses) < 2 else torch.stack(losses).mean()).mean()
