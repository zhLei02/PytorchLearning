import torch
import torchvision
from tqdm import tqdm
import argparse

from config.config import Config
from trainer.trainer import TrainAgent
from dataset.mnist_dataset import get_data_loader, train_dataset, test_dataset
from utils.utils import cycle

# import sys
# sys.argv = ['--exp_name','try']

parser = argparse.ArgumentParser()
parser.add_argument("--proj_dir", type=str, default="proj_log", help="path to project dir where models and logs are saved")
parser.add_argument("--exp_name", type=str, required=True, help="name of the experiment")
parser.add_argument("-bs", "--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--epochs", type=int, default=20, help="number of epochs")
parser.add_argument("--test", action="store_true", help="test mode") # "store_true" 意为默认为false，若指定该参数，则设为true
parser.add_argument("--continue", dest="cont", action="store_true", help="continue training from last saved checkpoint")
parser.add_argument("--ckpt", type=str, default="latest", required=False, help="checkpoint to load")
parser.add_argument("-g", "--gpu", type=int, default=0, help="gpu id")

args = parser.parse_args()

cfg = Config(args)

agent = TrainAgent(cfg)
if not args.test:
    # 训练
    if args.cont:
        agent.load_ckpt(args.ckpt)
    
    train_loader = get_data_loader(train_dataset, cfg)
    val_loader = get_data_loader(test_dataset, cfg)
    val_loader = cycle(val_loader)
    
    clock = agent.clock
    
    for epoch in range(clock.epoch, cfg.epochs):
        pbar = tqdm(train_loader)
        for b, data in enumerate(pbar):
            outputs, losses = agent.train_func(data)
            
            pbar.set_description("EPOCH[{}][{}]".format(epoch, b))
            pbar.set_postfix({k:v.item() for k,v in losses.items()})
            
            if clock.step % 5 == 0:
                data = next(val_loader)
                outputs, losses = agent.val_func(data)
            
            clock.tick()
        clock.tock()
        
        if clock.epoch % 5 == 0:
            agent.save_ckpt()
        
        agent.save_ckpt("latest")
else:
    # 测试
    val_loader = get_data_loader(test_dataset, cfg)
    agent.load_ckpt(args.ckpt)
    
    '''tensorboard 可视化'''
    images, labels = next(iter(val_loader))
    grid = torchvision.utils.make_grid(images)
    agent.test_tb.add_image("images", grid)
    agent.test_tb.add_text("labels", " ".join([str(x.item()) for x in labels]))
    agent.test_tb.add_graph(agent.model.to(torch.device('cpu')), images)
    agent.model.to(torch.device('cuda'))
    '''-----------------------------------'''
    
    pbar = tqdm(val_loader)
    
    total = 0
    correct = 0
    for b, data in enumerate(pbar):
        with torch.no_grad():
            outputs, losses = agent.val_func(data)
        
        prey = outputs.argmax(dim=1).detach().cpu()
        total += data[1].shape[0]
        correct += (prey == data[1]).sum().item() # item() 用于提取张量中的元素
    
    acc = correct / total
    print("Accuracy: {}".format(acc))