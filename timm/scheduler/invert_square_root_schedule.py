class FairseqLRScheduler(object):
    def __init__(self, optimizer):
        super().__init__()

        self.optimizer = optimizer
        self.best = None


    def state_dict(self):
        """Return the LR scheduler state dict."""
        return {"best": self.best}

    def load_state_dict(self, state_dict):
        """Load an LR scheduler state dict."""
        self.best = state_dict["best"]

    def step_begin_epoch(self, epoch):
        """Update the learning rate at the beginning of the given epoch."""
        pass

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        if val_loss is not None:
            if self.best is None:
                self.best = val_loss
            else:
                self.best = min(self.best, val_loss)

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        return self.optimizer.param_groups[0]['lr']

class InverseSquareRootSchedule(FairseqLRScheduler):
    """Decay the LR based on the inverse square root of the update number.

    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (``--warmup-init-lr``) until the configured
    learning rate (``--lr``). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.

    During warmup::

      lrs = torch.linspace(cfg.warmup_init_lr, cfg.lr, cfg.warmup_updates)
      lr = lrs[update_num]

    After warmup::

      decay_factor = cfg.lr * sqrt(cfg.warmup_updates)
      lr = decay_factor / sqrt(update_num)
    """

    def __init__(self, args, optimizer):
        super().__init__(optimizer)
        warmup_end_lr = args.lr
        self.args=args
        if args.warmup_init_lr < 0:
            args.warmup_init_lr = 0 if args.warmup_updates > 0 else warmup_end_lr

        # linearly warmup for the first cfg.warmup_updates
        self.lr_step = (warmup_end_lr - args.warmup_init_lr) / args.warmup_updates

        # then, decay prop. to the inverse square root of the update number
        self.decay_factor = warmup_end_lr * args.warmup_updates**0.5

        # initial learning rate
        self.lr = args.warmup_init_lr
        #self.optimizer.set_lr(self.lr)
        self.update_lr(self.lr)
    def update_lr(self,lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] =lr
        
    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.param_groups[0]['lr']

    def step_update(self, num_updates,metric=None):
        """Update the learning rate after each update."""
        if num_updates < self.args.warmup_updates:
            self.lr = self.args.warmup_init_lr + num_updates * self.lr_step
        else:
            self.lr = self.decay_factor * num_updates**-0.5
        #self.optimizer.set_lr(self.lr)
        self.update_lr(self.lr)
        return self.lr