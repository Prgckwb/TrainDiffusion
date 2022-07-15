import matplotlib.pyplot as plt

from config.options import parse_args_and_config
from scripts.utils import get_dataset


# utils.get_dataset のテスト
def test_get_dataset():
    args, config = parse_args_and_config()

    d1, d2 = get_dataset(args, config)
    img = d1.dataset[1][0]
    img = img.permute(1, 2, 0)
    img = img.numpy()
    plt.imsave("fuga.jpg", img)


if __name__ == '__main__':
    test_get_dataset()
