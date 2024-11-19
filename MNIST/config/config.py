import os
import time
import shutil
from utils.utils import ensure_dirs
import json

class Config(object):
    def __init__(self, args):
        self.exp_dir = os.path.join(args.proj_dir, args.exp_name, "MINST")
        current_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        self.log_dir = os.path.join(self.exp_dir, "log", current_time)
        self.model_dir = os.path.join(self.exp_dir, "model")
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.lr_step_size = 50
        self.train = not args.test
        self.grad_clip = None
        self.epochs = args.epochs
        
        if (not args.test) and args.cont is not True and os.path.exists(self.exp_dir):
            response = input("The experiment dir already exists, overwrite? (y/n)")
            if response != 'y':
                exit()
            shutil.rmtree(self.exp_dir)
        
        ensure_dirs([self.log_dir, self.model_dir])
        if not args.test:
            with open("{}/config.txt".format(self.exp_dir),'w') as f:
                json.dump(args.__dict__, f, indent=2)