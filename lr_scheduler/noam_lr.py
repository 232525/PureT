import torch
from lib.config import cfg

class NoamLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, model_size, factor, warmup, last_epoch=-1):
        # last_epoch用于恢复学习率
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        super(NoamLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # LR设置与base_lr无关，即默认从0开始
        return [
            self.factor * \
            (self.model_size ** (-0.5) *
            min((self.last_epoch + 1) ** (-0.5), (self.last_epoch + 1) * self.warmup ** (-1.5)))
            for base_lr in self.base_lrs
        ]