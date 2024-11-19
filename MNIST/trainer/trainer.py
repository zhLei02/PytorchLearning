from .base import BaseTrainer
from torch import nn
from torch import optim
from model.CNN import CNN

class TrainAgent(BaseTrainer):
    def build_net(self, cfg):
        self.model = CNN()
        self.model.to(self.device)
    
    def set_loss_function(self, cfg):
        self.loss_fn = nn.CrossEntropyLoss()
    
    def set_optimizer(self, cfg):
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=cfg.lr_step_size)
    
    def forward(self, data):
        x, y = data
        x = x.to(self.device)
        y = y.to(self.device)
        outputs = self.model(x)
        losses = {
            "loss": self.loss_fn(outputs, y)
        }
        return outputs, losses