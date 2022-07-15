from config.options import parse_args_and_config
from scripts.diffusion import Diffusion


def main():
    args, config = parse_args_and_config()

    runner = Diffusion(args, config)
    if args.sample:
        runner.sample()
    elif args.test:
        runner.test()
    else:
        runner.train()

    return 0


if __name__ == '__main__':
    main()
