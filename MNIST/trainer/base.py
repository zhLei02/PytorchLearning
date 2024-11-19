import torch
from torch import optim
from tensorboardX import SummaryWriter
from abc import abstractmethod
import os
from typing import Tuple

'''
训练器
'''
class BaseTrainer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        
        self.log_dir = cfg.log_dir
        self.model_dir = cfg.model_dir
        self.clock = TrainClock()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = cfg.batch_size
        
        self.build_net(cfg)
        
        self.set_loss_function(cfg)
        
        self.set_optimizer(cfg)
        
        self.train_tb = SummaryWriter(os.path.join(self.log_dir, "train"))
        self.test_tb = SummaryWriter(os.path.join(self.log_dir, "test"))
    
    @abstractmethod
    def build_net(self, cfg):
        raise NotImplementedError

    def set_loss_function(self, cfg):
        pass
    
    def set_optimizer(self, cfg):
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=cfg.lr_step_size)
    
    def save_ckpt(self, name = None):
        if name is None:
            name = "epoch{}.pth".format(self.clock.epoch)
        else:
            name = name + ".pth"
        save_path = os.path.join(self.model_dir, name)
        
        if self.device == torch.device("cuda"):
            model = self.model.cpu()
        
        torch.save({
            'clock': self.clock.make_ckpt(),
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }, save_path)
        
        self.model.to(self.device)
    
    def load_ckpt(self, name = None):
        name = 'latest.pth' if name == 'latest' else "epoch{}.pth".format(name)
        load_path = os.path.join(self.model_dir, name)
        if not os.path.exists(load_path):
            raise ValueError("No such checkpoint: {}".format(load_path))
        
        checkpoint = torch.load(load_path)
        self.clock.restore_ckpt(checkpoint['clock'])
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
    
    @abstractmethod
    def forward(self, data):
        raise NotImplementedError
    
    """
    更新网络参数
    """
    def update_network(self, loss_dict:dict):
        loss = sum(loss_dict.values())
        self.optimizer.zero_grad() # 清空梯度
        loss.backward() # 反向传播
        if self.cfg.grad_clip is not None:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step() # 更新参数
    
    """
    更新学习率
    """
    def update_learning_rate(self):
        self.train_tb.add_scalar("lr", self.optimizer.param_groups[-1]['lr'], self.clock.epoch)
        self.scheduler.step()
    
    """
    记录损失
    """
    def record_losses(self, loss_dict, mode = "train"):
        loss_values = {k:v.item() for k, v in loss_dict.items()}
        
        tb = self.train_tb if mode == "train" else self.test_tb
        for k, v in loss_values.items():
            tb.add_scalar(k, v, self.clock.step)
        
    def train_func(self,data):
        self.model.train() # 设置为训练模式
        
        outputs, losses = self.forward(data)
        
        self.update_network(losses)
        if self.clock.step % 10 == 0:
            self.record_losses(losses, "train")
        
        return outputs, losses
    
    def val_func(self, data)-> Tuple[torch.Tensor, dict]:
        self.model.eval()
        with torch.no_grad():
            outputs, losses = self.forward(data)
        
        self.record_losses(losses, "test")
        return outputs, losses
        
class TrainClock(object):
    def __init__(self):
        self.epoch = 0
        self.iter = 0
        self.step = 0

    def tick(self):
        self.step += 1
        self.iter += 1
    
    def tock(self):
        self.epoch += 1
        self.iter = 0
    
    def make_ckpt(self):
        return {
            'epoch': self.epoch,
            'iter': self.iter,
            'step': self.step
        }
    
    def restore_ckpt(self, ckp):
        self.epoch = ckp['epoch']
        self.iter = ckp['iter']
        self.step = ckp['step']