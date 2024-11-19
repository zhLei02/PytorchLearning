from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config.config import Config

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST('./data/mnist', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data/mnist', train=False, transform=transform)

def get_data_loader(dataset:datasets, config:Config, shuffle:bool = None) -> DataLoader:
    is_shuffle = config.train if shuffle is None else shuffle
    
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=is_shuffle)
    return dataloader