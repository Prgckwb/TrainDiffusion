import numpy as np
import torch

# ArgparseとConfigファイルから逃げるための横着クラス
# TODO: なんとかする
class Arguments:
    def __init__(self):
        self.seed = 1234
        self.exp = "exp"
        self.comment = ""
        self.doc = "hoge"
        self.verbose = "info"
        self.test = False
        self.sample = False
        self.fid = None
        self.interpolation = None
        self.resume_training = None
        self.image_folder = "images"
        self.ni = None
        self.sample_type = "generalized"
        self.skip_type = "uniform"
        self.timesteps = 1000
        self.eta = 0.0
        self.sequence = None


class Data:
    def __init__(self):
        self.image_size = 64
        self.channels = 3
        self.logit_transform = False
        self.uniform_dequantization = False
        self.gaussian_dequantization = False
        self.random_flip = True
        self.rescaled = True
        self.num_workers = 4


class Model:
    def __init__(self):
        self.type = "simple"
        self.in_channels = 3
        self.out_ch = 3
        self.ch = 128
        self.ch_mult = [1, 2, 2, 2]
        self.num_res_block = 2
        self.atte_resolutions = [16, ]
        self.dropout = 0.1
        self.var_type = "fixedlarge"
        self.ema_rate = 0.9999
        self.ema = True
        self.resamp_with_conv = True


class Diffusion:
    def __init__(self):
        self.beta_schedule = "linear"
        self.beta_start = 0.0001
        self.beta_end = 0.02
        self.num_diffusion_timestep = 1000


class Training:
    def __init__(self):
        self.batch_size = 128
        self.n_epochs = 10000
        self.n_iters = 5000000
        self.snapshot_freq = 5000
        self.validation_freq = 2000


class Sampling:
    def __init__(self):
        self.batch_size = 64
        self.last_only = True


class Optimizer:
    def __init__(self):
        self.weight_decay = 0.000
        self.optimizer = "Adam"
        self.lr = 0.0002
        self.beta1 = 0.9
        self.amsgrad = False
        self.eps = 0.00000001
        self.grad_clip = 1.0


class Config:
    def __init__(self):
        self.data = Data()
        self.model = Model()
        self.diffusion = Diffusion()
        self.training = Training()
        self.sampling = Sampling()
        self.optim = Optimizer()


def parse_args_and_config():
    args = Arguments()
    config = Config()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    return args, config
