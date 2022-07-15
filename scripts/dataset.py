import glob

import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms


class MyDataset(data.Dataset):
    def __init__(self, image_dir, transform=None):
        self.transform = transform
        self.imgdir = image_dir
        self.image_list = sorted(glob.glob(self.imgdir + "/*"))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img = Image.open(self.image_list[index]).convert("RGB")
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
    img_size = (config.data.image_size, config.data.image_size)
    transform = transforms.Compose([
        transforms.Resize(img_size),
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
