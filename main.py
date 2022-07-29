import argparse
import re
from pathlib import Path

from config.options import Arguments, Config
from config.options import parse_args_and_config
from scripts.diffusion import Diffusion


# Specify the mode with command line arguments
def init_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", required=True, type=str)
    parser.add_argument("-r", "--resume_num", default=None, type=int)
    parser.add_argument("-s", "--size", type=int)
    parser.add_argument("-n", "--checkpoint_num", type=int)
    parser.add_argument("-b", "--train_batch_size", type=int, default=128)

    cmdline_args = parser.parse_args()
    args, config = parse_args_and_config(cmdline_args)

    return args, config


# checkpointの中で一番新しい番号のものを取り出す
def get_max_ckpt_num(args: Arguments, config: Config):
    arg_num = args.sample_ckpt_num
    if arg_num is None:
        logdir_path = Path(f"{args.log_path}")
        paths = logdir_path.glob("*.pth")

        # 一番新しいcheckpointsの取得
        p = re.compile(r'\d+')

        # 正規表現でパスの数字部分だけ取り出す
        def get_ckpt_num(path):
            return int(p.findall(str(path))[-1])

        ckpt_paths = sorted(paths, key=get_ckpt_num)
        arg_num = ckpt_paths[-1].stem.split("_")[-1]

    return arg_num


def main():
    args, config = init_options()

    runner = Diffusion(args, config)

    if args.sample:
        ckpt_num = get_max_ckpt_num(args, config)
        runner.sample(ckpt_num=ckpt_num)
    elif args.test:
        runner.test()
    else:
        runner.train()

    return 0


if __name__ == '__main__':
    main()
