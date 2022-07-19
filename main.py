import argparse

from config.options import parse_args_and_config
from scripts.diffusion import Diffusion


# Specify the mode with command line arguments
def init_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", required=True, type=str)
    # parser.add_argument("-n", "--resume_num", default=None, type=int)
    parser.add_argument("-s", "--size", type=int, default=64)
    parser.add_argument("-n", "--checkpoint_num", type=int)
    parser.add_argument("-b", "--train_batch_size", type=int, default=128)

    cmdline_args = parser.parse_args()
    mode = cmdline_args.mode

    args, config = parse_args_and_config(cmdline_args.size)

    config.training.batch_size = cmdline_args.train_batch_size

    if mode == "test":
        args.test = True
        args.sample = False
    elif mode == "sample":
        args.test = False
        args.sample = True
        args.sample_ckpt_num = cmdline_args.checkpoint_num
    elif mode == "train":
        args.test = False
        args.sample = False

    return args, config


def main():
    args, config = init_options()
    print(f"[DEBUG] train_batch_size: {config.training.batch_size}")

    runner = Diffusion(args, config)

    if args.sample:
        ckpt_num = args.sample_ckpt_num
        if ckpt_num is None:
            print("Please select ckpt_num")
            return
        runner.sample(ckpt_num=ckpt_num)
    elif args.test:
        runner.test()
    else:
        runner.train()

    return 0


if __name__ == '__main__':
    main()
