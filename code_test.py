from typing import OrderedDict

import torch
from torch.nn import DataParallel

from config.options import parse_args_and_config
from main import init_cmdline_arguments
from models.diffusion_model import Model
from scripts.dataset import get_dataset
from scripts.diffusion import Diffusion


# utils.get_dataset のテスト
def test_get_dataset():
    args, config = parse_args_and_config()

    d1, d2 = get_dataset(args, config)

    for i in range(100):
        img = d1.dataset[i][0]
        print(i, img.shape)
    # img = img.permute(1, 2, 0)
    # img = img.numpy()
    # plt.imsave("fuga.jpg", img)


def test_generate():
    args, config = parse_args_and_config()
    args = init_cmdline_arguments(args)
    runner = Diffusion(args, config)

    model = Model(config)
    model = DataParallel(model)
    # pthはstates = [model.state_dict(),optimizer.state_dict(),epoch,step]
    # のリストになっている
    ckpt: OrderedDict = torch.load("log/ckpt_10000.pth")[0]
    # for key in ckpt.keys():
    #     print(key)
    model = model.load_state_dict(ckpt)
    # print(model)


if __name__ == '__main__':
    # test_get_dataset()
    test_generate()
