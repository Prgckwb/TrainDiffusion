import dataclasses

import numpy as np
import torch


# ArgparseとConfigファイルから逃げるための横着クラス
# TODO: なんとかする
@dataclasses.dataclass
class Arguments:
    # test: False & sample: False -> train mode
    test = False
    sample = False

    seed = 1234
    exp = "exp"
    comment = ""
    doc = "hoge"
    verbose = "info"
    resume_training = None
    ni = None
    sample_type = "generalized"
    skip_type = "uniform"
    timesteps = 1000
    eta = 0.0

    # sampleの仕方などなど
    sequence = None
    interpolation = None
    fid = None

    sample_options = ["sequence", "interpolation", "fid"]
    sample_mode = sample_options[0]

    image_folder = "images"
    log_path = "logs"

    use_pretrained = True


@dataclasses.dataclass
class Data:
    # image_sizeを変えるとload_state_dictでエラーが出るかも
    image_size = 64
    channels = 3
    logit_transform = False
    uniform_dequantization = False
    gaussian_dequantization = False
    random_flip = True
    rescaled = True
    num_workers = 4


@dataclasses.dataclass
class Model:
    type = "simple"
    in_channels = 3
    out_ch = 3
    ch = 128
    ch_mult = [1, 2, 2, 2]
    num_res_blocks = 2
    attn_resolutions = [16, ]
    dropout = 0.1
    var_type = "fixedlarge"
    ema_rate = 0.9999
    ema = True
    resamp_with_conv = True


@dataclasses.dataclass
class Diffusion:
    beta_schedule = "linear"
    beta_start = 0.0001
    beta_end = 0.02
    num_diffusion_timesteps = 1000


@dataclasses.dataclass
class Training:
    batch_size = 64
    n_epochs = 10000
    n_iters = 5000000
    snapshot_freq = 5000
    validation_freq = 2000


@dataclasses.dataclass
class Sampling:
    batch_size = 64
    last_only = True


@dataclasses.dataclass
class Optimizer:
    weight_decay = 0.000
    optimizer = "Adam"
    lr = 0.0002
    beta1 = 0.9
    amsgrad = False
    eps = 0.00000001
    grad_clip = 1.0


@dataclasses.dataclass
class Config:
    data = Data()
    model = Model()
    diffusion = Diffusion()
    training = Training()
    sampling = Sampling()
    optim = Optimizer()


def parse_args_and_config(img_size):
    args = Arguments()
    config = Config()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.sample_mode == args.sample_options[0]:
        args.sequence = True
    elif args.sample_mode == args.sample_options[1]:
        args.interpolation = True
    elif args.sample_mode == args.sample_options[2]:
        args.fid = True

    print(args.sample_mode)
    config.data.image_size = img_size
    args.image_folder = f"{args.image_folder}/size_{img_size}/{args.sample_mode}"
    args.log_path = f"{args.log_path}/size_{img_size}"

    if img_size == 64:
        args.log_path = "log"

    return args, config


if __name__ == '__main__':
    a, c = parse_args_and_config(128)
    print(a)
