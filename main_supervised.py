import tensorflow as tf
from math import ceil
from model.networks import Supervised_UNet
from options.baseOptions import BaseOptions
from data.DB import DB as myDB

opt = BaseOptions().parse()
sdir = opt.savepath + '/' + opt.name + '/'
opt.log_dir = sdir + 'log_dir'
opt.ckpt_dir = sdir + 'ckpt_dir'


config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)

model = Supervised_UNet(sess, opt)

if opt.is_training:
    DB_train_A = myDB(opt, 'train', 'A')
    DB_train_B = myDB(opt, 'train', 'B')
    opt.nStep_train = ceil(DB_train_A.len / opt.batch_size)
    model.train(opt, DB_train_A, DB_train_B)
    opt.is_training = False

DB_test_B = myDB(opt, 'test', 'B')
opt.nStep_test = ceil(DB_test_B.len / 1)
model.test(opt, DB_test_B)
