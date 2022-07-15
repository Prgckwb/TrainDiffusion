import glob

import numpy as np
import torch.optim as optim
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils import data
from torchvision import transforms


def get_optimizer(config, parameters):
    if config.optim.optimizer == 'Adam':
        return optim.Adam(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay,
                          betas=(config.optim.beta1, 0.999), amsgrad=config.optim.amsgrad,
                          eps=config.optim.eps)
    elif config.optim.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'SGD':
        return optim.SGD(parameters, lr=config.optim.lr, momentum=0.9)
    else:
        raise NotImplementedError(
            'Optimizer {} not understood.'.format(config.optim.optimizer))


class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )


class MyDataset(data.Dataset):
    def __init__(self, image_dir, transform=None):
        self.transform = transform
        self.imgdir = image_dir
        self.image_list = sorted(glob.glob(self.imgdir + "/*"))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img = Image.open(self.image_list[index])
        img = self.transform(img)
        label = 0

        return img, label


def get_dataset(args, config):
    if config.data.random_flip is False:
        tran_transform = test_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor()]
        )
    else:
        tran_transform = transforms.Compose(
            [
                transforms.Resize(config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor()]
        )

    # TODO: random_flipに対応する？
    transform = transforms.Compose([
        transforms.Resize(config.data.image_size),
        transforms.ToTensor(),
    ])

    # TODO: ハードコーディングを避ける！！！
    dataset = MyDataset("../data/ramen", transform=transform)

    # datasetの分割
    num_items = len(dataset)
    indices = list(range(num_items))
    random_state = np.random.get_state()
    np.random.seed(2019)
    np.random.shuffle(indices)
    np.random.set_state(random_state)
    train_indices, test_indices = (
        indices[: int(num_items * 0.9)],
        indices[int(num_items * 0.9):],
    )
    test_dataset = data.Subset(dataset, test_indices)
    dataset = data.Subset(dataset, train_indices)

    return dataset, test_dataset
