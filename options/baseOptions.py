import argparse
import os


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataroot_A', type=str, default='../Data/cycleGAN_MR/single/full', help='path for dataset (full)')
        self.parser.add_argument('--dataroot_B', type=str, default='../Data/cycleGAN_MR/single/down4_2D', help='path for dataset (down)')
        self.parser.add_argument('--nEpoch', type=int, default=100, help='number of epoch iteration')
        self.parser.add_argument('--decay_epoch', type=int, default=100, help='start point of lr decay')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        self.parser.add_argument('--disp_div_N', type=int, default=30, help='display N times per epoch')
        self.parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        self.parser.add_argument('--nC', type=int, default=1, help='number of coils')
        self.parser.add_argument('--nX', type=int, default=640, help='width of image')
        self.parser.add_argument('--ngf', type=int, default=32, help='number of filters of generator')
        self.parser.add_argument('--ndf', type=int, default=64, help='number of filters of discriminator')
        self.parser.add_argument('--norm', type=str, default='instance', help='normalization method (instance, batch)')
        self.parser.add_argument('--cyc_lambda', type=float, default=0.5, help='lambda for cycle loss')
        self.parser.add_argument('--iden_lambda', type=float, default=0, help='lambda for identity loss')
        self.parser.add_argument('--GP_lambda', type=float, default=5, help='lambda for gradient penalty loss')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='moment, 0 to 1')
        self.parser.add_argument('--beta2', type=float, default=0.9, help='moment, 0 to 1')
        self.parser.add_argument('--pool_size', type=int, default=50, help='size of image pool')
        self.parser.add_argument('--is_training', action='store_true', help='True if training')
        self.parser.add_argument('--use_lsgan', action='store_true', help='True if use lsgan')
        self.parser.add_argument('--use_wgan', action='store_true', help='True if use wgan')
        self.parser.add_argument('--use_identity', action='store_true', help='use identity loss')
        self.parser.add_argument('--generator', type=str, default='unet', help='type of generator, unet, unet_residual')
        self.parser.add_argument('--discriminator', type=str, default='basic', help='type of discriminator')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu_ids: e.g. 0  0,1,2')
        self.parser.add_argument('--savepath', type=str, default='./Results', help='path for saving results')
        self.parser.add_argument('--name', type=str, default='experiment name', help='name of experiment')
        self.parser.add_argument('--small_DB', action='store_true', help='use small DB')
        self.parser.add_argument('--n_critic', type=int, default=5, help='update discriminator n times per 1 updates of generator')
        self.parser.add_argument('--DSrate', type=int, default=4, help='downsampling rate')

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        args = vars(self.opt)

        print('---------- Options ----------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('------------ End ------------')

        expr_dir = os.path.join(self.opt.savepath, self.opt.name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('---------- Options ----------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('------------ End ------------')
        opt_file.close()
        return self.opt
