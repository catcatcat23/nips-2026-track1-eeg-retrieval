"""Step warmup followed by exactly N non-improving validations per LR drop."""
import torch

class WarmupPlateau:
    def __init__(self, optimizer, warmup_steps, cfg):
        self.optimizer=optimizer
        self.base_lrs=[g['lr'] for g in optimizer.param_groups]
        self.warmup_steps=warmup_steps
        self.steps=0
        self.plateau=torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,mode='max',factor=cfg['plateau_factor'],
            patience=cfg['plateau_patience']-1,threshold=0.,threshold_mode='abs',
            min_lr=[lr*cfg['min_lr']/cfg['lr'] for lr in self.base_lrs])
        self._warmup()
    def _warmup(self):
        for g,lr in zip(self.optimizer.param_groups,self.base_lrs):
            g['lr']=lr*min(1.,(self.steps+1)/max(1,self.warmup_steps))
    def step(self):
        self.steps+=1
        if self.steps<=self.warmup_steps:self._warmup()
    def validation_step(self,score):
        if self.steps>=self.warmup_steps:self.plateau.step(score)
    def get_last_lr(self):return [g['lr'] for g in self.optimizer.param_groups]
    def state_dict(self):
        return {'steps':self.steps,'base_lrs':self.base_lrs,'warmup_steps':self.warmup_steps,'plateau':self.plateau.state_dict()}
