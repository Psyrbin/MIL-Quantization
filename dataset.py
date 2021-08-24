import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

class MapDataset(torch.utils.data.Dataset):
    """
    Given a dataset, creates a dataset which applies a mapping function
    to its items.

    Note that data is not cloned/copied from the initial dataset.
    """

    def __init__(self, dataset, map_fn):
        self.dataset = dataset
        self.map = map_fn

    def __getitem__(self, index):
        x = self.map(self.dataset[index][0])
        y = self.dataset[index][1]
        return x, y

    def __len__(self):
        return len(self.dataset)

def get_cifar10(transform_train=True):
    cifar10_train = torchvision.datasets.CIFAR10(root='./cifar10',
                                                 train=True,
                                                 transform=None,
                                                 target_transform=None,
                                                 download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='./cifar10',
                                                 train=False,
                                                 transform=None,
                                                 target_transform=None,
                                                 download=True)



    train_transform = transforms.Compose([
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                              mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                    ])

    test_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                              mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                    ])


    train_dataset, val_dataset = torch.utils.data.random_split(cifar10_train, [45000, 5000], generator=torch.Generator().manual_seed(42))
    if transform_train:
        train_dataset = MapDataset(train_dataset, train_transform)
    else:
        train_dataset = MapDataset(train_dataset, test_transform)
    val_dataset = MapDataset(val_dataset, test_transform)
    test_dataset = MapDataset(test_dataset, test_transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=False)#, num_workers=10, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, drop_last=False)#, num_workers=10, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, drop_last=False)#, num_workers=10, pin_memory=True)

    return train_loader, val_loader, test_loader
