import argparse

from config.options import parse_args_and_config
from scripts.diffusion import Diffusion


# Specify the mode with command line arguments
def init_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", required=True, type=str)
    # parser.add_argument("-n", "--resume_num", default=None, type=int)
    parser.add_argument("-s", "--size", type=int, default=64)
    parser.add_argument("-br", "--train_batch_size", type=int, default=128)

    cmdline_args = parser.parse_args()
    mode = cmdline_args.mode

    args, config = parse_args_and_config(cmdline_args.size)

    if mode == "test":
        args.test = True
        args.sample = False
    elif mode == "sample":
        args.test = False
        args.sample = True
    elif mode == "train":
        args.test = False
        args.sample = False

    return args, config


def main():
    args, config = init_options()

    runner = Diffusion(args, config)

    if args.sample:
        runner.sample(ckpt_num=40000)
    elif args.test:
        runner.test()
    else:
        runner.train()

    return 0


if __name__ == '__main__':
    main()
