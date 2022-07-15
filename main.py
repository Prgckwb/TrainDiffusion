import argparse

from config.options import Arguments
from config.options import parse_args_and_config
from scripts.diffusion import Diffusion


# Specify the mode with command line arguments
def init_cmdline_arguments(args: Arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", required=True, type=str)
    # parser.add_argument("-n", "--resume_num", default=None, type=int)

    cmdline_args = parser.parse_args()
    mode = cmdline_args.mode

    if mode == "test":
        args.test = True
        args.sample = False
    elif mode == "sample":
        args.test = False
        args.sample = True
    elif mode == "train":
        args.test = False
        args.sample = False

    return args


def main():
    args, config = parse_args_and_config()
    args = init_cmdline_arguments(args)

    runner = Diffusion(args, config)

    if args.sample:
        runner.sample(resume_num=45000)
    elif args.test:
        runner.test()
    else:
        runner.train()

    return 0


if __name__ == '__main__':
    main()
