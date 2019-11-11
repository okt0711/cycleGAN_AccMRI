import os
import time
import tensorflow as tf
from scipy import io as sio
from math import ceil
from model.generator import UnetGenerator, FFTGenerator
from model.discriminator import Discriminator
from model.criterion import *
from utils.utils import *
from ipdb import set_trace as st

dtype = tf.float32


class CycleGAN_AccMRI:
    def __init__(self, sess, opt):
        GPU = '/device:GPU:' + str(opt.gpu_ids[0])
        print(GPU)

        self.GPU = GPU
        self.sess = sess
        self.nX = opt.nX
        self.nC = opt.nC
        self.cyc_lambda = opt.cyc_lambda
        self.iden_lambda = opt.iden_lambda
        self.GP_lambda = opt.GP_lambda
        self.beta1 = opt.beta1
        self.beta2 = opt.beta2
        self.opt = opt
        self.generator = opt.generator
        self.discriminator = opt.discriminator
        self.use_identity = opt.use_identity
        self.use_wgan = opt.use_wgan
        self.n_critic = opt.n_critic

        if opt.use_lsgan:
            self.criterionGAN = mse_criterion
            self.use_sigmoid = False
        elif opt.use_wgan:
            self.use_sigmoid = False
        else:
            self.criterionGAN = sce_criterion
            self.use_sigmoid = True

        self._build_model()
        self.saver = tf.train.Saver()
        self.pool = ImagePool(opt.pool_size)

    def _build_model(self):
        with tf.device(self.GPU):
            if self.generator == 'unet':
                self.generatorA2B = UnetGenerator(self.opt, name='generatorA2B')
                self.generatorB2A = UnetGenerator(self.opt, name='generatorB2A')
            elif self.generator == 'unet_residual':
                self.generatorA2B = UnetGenerator(self.opt, res=True, name='generatorA2B')
                self.generatorB2A = UnetGenerator(self.opt, res=True, name='generatorB2A')
            else:
                st()

            if self.discriminator == 'basic':
                self.discriminatorA = Discriminator(self.opt, name='discriminatorA', use_sigmoid=self.use_sigmoid)
                self.discriminatorB = Discriminator(self.opt, name='discriminatorB', use_sigmoid=self.use_sigmoid)
            else:
                st()

            self.real_A = tf.placeholder(dtype, [1, None, self.nX, self.nC * 2], name='real_A')
            self.real_B = tf.placeholder(dtype, [1, None, self.nX, self.nC * 2], name='real_B')

            self.fake_B = self.generatorA2B(self.real_A)
            self.fake_A = self.generatorB2A(self.real_B)
            self.recon_B = self.generatorA2B(self.fake_A)
            self.recon_A = self.generatorB2A(self.fake_B)

            self.iden_B = self.generatorA2B(self.real_B)
            self.iden_A = self.generatorB2A(self.real_A)

            self.DA_fake = self.discriminatorA(self.fake_A)
            self.DB_fake = self.discriminatorB(self.fake_B)

            if self.use_wgan:
                self.gan_loss = -tf.reduce_mean(self.DB_fake) - tf.reduce_mean(self.DA_fake)
                self.cycle_loss = mse_criterion(self.real_A, self.recon_A) + mse_criterion(self.real_B, self.recon_B)
                self.g_loss_a2b = -tf.reduce_mean(self.DB_fake) + self.cyc_lambda * self.cycle_loss
                self.g_loss_b2a = -tf.reduce_mean(self.DA_fake) + self.cyc_lambda * self.cycle_loss
                self.g_loss = self.gan_loss + self.cyc_lambda * self.cycle_loss

            else:
                self.gan_loss = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) \
                                + self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake))
                self.cycle_loss = mae_criterion(self.real_A, self.recon_A) + mae_criterion(self.real_B, self.recon_B)
                self.g_loss_a2b = self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) \
                                  + self.cyc_lambda * self.cycle_loss
                self.g_loss_b2a = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) \
                                  + self.cyc_lambda * self.cycle_loss
                self.g_loss = self.gan_loss + self.cyc_lambda * self.cycle_loss

            if self.use_identity:
                self.iden_loss = mae_criterion(self.real_A, self.iden_A) + mae_criterion(self.real_B, self.iden_B)
                self.g_loss_a2b = self.g_loss_a2b + self.iden_lambda * self.iden_loss
                self.g_loss_b2a = self.g_loss_b2a + self.iden_lambda * self.iden_loss
                self.g_loss = self.g_loss + self.iden_lambda * self.iden_loss

            self.fake_A_sample = tf.placeholder(dtype, [1, None, self.nX, self.nC * 2], name='fake_A_sample')
            self.fake_B_sample = tf.placeholder(dtype, [1, None, self.nX, self.nC * 2], name='fake_B_sample')
            self.DA_real = self.discriminatorA(self.real_A)
            self.DB_real = self.discriminatorB(self.real_B)
            self.DA_fake_sample = self.discriminatorA(self.fake_A_sample)
            self.DB_fake_sample = self.discriminatorB(self.fake_B_sample)

            if self.use_wgan:
                self.da_loss_real = -tf.reduce_mean(self.DA_real)
                self.da_loss_fake = tf.reduce_mean(self.DA_fake_sample)
                self.da_loss_GP = gradient_penalty(self.fake_A_sample, self.real_A, 1, self.discriminatorA)
                self.da_loss = (self.da_loss_real + self.da_loss_fake + self.GP_lambda * self.da_loss_GP) / 2
                self.db_loss_real = -tf.reduce_mean(self.DB_real)
                self.db_loss_fake = tf.reduce_mean(self.DB_fake_sample)
                self.db_loss_GP = gradient_penalty(self.fake_B_sample, self.real_B, 1, self.discriminatorB)
                self.db_loss = (self.db_loss_real + self.db_loss_fake + self.GP_lambda * self.db_loss_GP) / 2
                self.d_loss = self.da_loss + self.db_loss
            else:
                self.da_loss_real = self.criterionGAN(self.DA_real, tf.ones_like(self.DA_real))
                self.da_loss_fake = self.criterionGAN(self.DA_fake_sample, tf.zeros_like(self.DA_fake_sample))
                self.da_loss = (self.da_loss_real + self.da_loss_fake) / 2
                self.db_loss_real = self.criterionGAN(self.DB_real, tf.ones_like(self.DB_real))
                self.db_loss_fake = self.criterionGAN(self.DB_fake_sample, tf.zeros_like(self.DB_fake_sample))
                self.db_loss = (self.db_loss_real + self.db_loss_fake) / 2
                self.d_loss = self.da_loss + self.db_loss

            self.gan_loss_sum = tf.summary.scalar('generator/gan_loss', self.gan_loss)
            self.cycle_loss_sum = tf.summary.scalar('generator/cycle_loss', self.cycle_loss)
            self.g_loss_a2b_sum = tf.summary.scalar('generator/g_loss_a2b', self.g_loss_a2b)
            self.g_loss_b2a_sum = tf.summary.scalar('generator/g_loss_b2a', self.g_loss_b2a)
            self.g_loss_sum = tf.summary.scalar('generator/g_loss', self.g_loss)

            if self.use_identity:
                self.iden_loss_sum = tf.summary.scalar('generator/identity_loss', self.iden_loss)
                self.g_sum = tf.summary.merge([self.gan_loss_sum, self.cycle_loss_sum, self.iden_loss_sum,
                                               self.g_loss_a2b_sum, self.g_loss_b2a_sum, self.g_loss_sum])
            else:
                self.g_sum = tf.summary.merge([self.gan_loss_sum, self.cycle_loss_sum,
                                               self.g_loss_a2b_sum, self.g_loss_b2a_sum, self.g_loss_sum])

            self.da_loss_real_sum = tf.summary.scalar('discriminator/da_loss_real', self.da_loss_real)
            self.da_loss_fake_sum = tf.summary.scalar('discriminator/da_loss_fake', self.da_loss_fake)
            self.da_loss_sum = tf.summary.scalar('discriminator/da_loss', self.da_loss)
            self.db_loss_real_sum = tf.summary.scalar('discriminator/db_loss_real', self.db_loss_real)
            self.db_loss_fake_sum = tf.summary.scalar('discriminator/db_loss_fake', self.db_loss_fake)
            self.db_loss_sum = tf.summary.scalar('discriminator/db_loss', self.db_loss)
            self.d_loss_sum = tf.summary.scalar('discriminator/d_loss', self.d_loss)

            if self.use_wgan:
                self.da_loss_GP_sum = tf.summary.scalar('discriminator/da_loss_GP', self.da_loss_GP)
                self.db_loss_GP_sum = tf.summary.scalar('discirminator/db_loss_GP', self.db_loss_GP)
                self.d_sum = tf.summary.merge([self.da_loss_real_sum, self.da_loss_fake_sum, self.da_loss_GP_sum,
                                               self.da_loss_sum, self.db_loss_real_sum, self.db_loss_fake_sum,
                                               self.db_loss_GP_sum, self.db_loss_sum, self.d_loss_sum])
            else:
                self.d_sum = tf.summary.merge([self.da_loss_real_sum, self.da_loss_fake_sum, self.da_loss_sum,
                                               self.db_loss_real_sum, self.db_loss_fake_sum, self.db_loss_sum,
                                               self.d_loss_sum])

            self.scale_A = tf.placeholder(dtype, [1, None, self.nX, self.nC * 2], name='scale_tensor_A')
            self.scale_B = tf.placeholder(dtype, [1, None, self.nX, self.nC * 2], name='scale_tensor_B')

            self.real_A_ssos_sum = tf.summary.image('ssos/real_full', tf_imgri2ssos(self.real_A * self.scale_A),
                                                    max_outputs=1)
            self.real_B_ssos_sum = tf.summary.image('ssos/real_down', tf_imgri2ssos(self.real_B * self.scale_B),
                                                    max_outputs=1)
            self.fake_A_ssos_sum = tf.summary.image('ssos/fake_full', tf_imgri2ssos(self.fake_A * self.scale_B),
                                                    max_outputs=1)
            self.fake_B_ssos_sum = tf.summary.image('ssos/fake_down', tf_imgri2ssos(self.fake_B * self.scale_A),
                                                    max_outputs=1)
            self.recon_A_ssos_sum = tf.summary.image('ssos/recon_full', tf_imgri2ssos(self.recon_A * self.scale_A),
                                                     max_outputs=1)
            self.recon_B_ssos_sum = tf.summary.image('ssos/recon_down', tf_imgri2ssos(self.recon_B * self.scale_B),
                                                     max_outputs=1)
            self.ssos_sum = tf.summary.merge([self.real_A_ssos_sum, self.real_B_ssos_sum,
                                              self.fake_A_ssos_sum, self.fake_B_ssos_sum,
                                              self.recon_A_ssos_sum, self.recon_B_ssos_sum])

            self.test_real_A = tf.placeholder(dtype, [1, None, self.nX, self.nC * 2], name='test_real_A')
            self.test_real_B = tf.placeholder(dtype, [1, None, self.nX, self.nC * 2], name='test_real_B')

            self.test_fake_B = self.generatorA2B(self.test_real_A)
            self.test_fake_A = self.generatorB2A(self.test_real_B)

            self.test_fake_img_B = tf.squeeze(tf_ri2comp(self.test_fake_B * self.scale_A))
            self.test_fake_img_A = tf.squeeze(tf_ri2comp(self.test_fake_A * self.scale_B))

            self.lr = tf.placeholder(dtype, None, name='learning_rate')
            self.lr_sum = tf.summary.scalar('learning_rate', self.lr)

            self.ga2b_optim = tf.train.AdamOptimizer(self.lr, beta1=self.beta1, beta2=self.beta2). \
                minimize(self.g_loss_a2b, var_list=self.generatorA2B.variables)
            self.gb2a_optim = tf.train.AdamOptimizer(self.lr, beta1=self.beta1, beta2=self.beta2). \
                minimize(self.g_loss_b2a, var_list=self.generatorB2A.variables)
            self.da_optim = tf.train.AdamOptimizer(self.lr, beta1=self.beta1, beta2=self.beta2). \
                minimize(self.da_loss, var_list=self.discriminatorA.variables)
            self.db_optim = tf.train.AdamOptimizer(self.lr, beta1=self.beta1, beta2=self.beta2). \
                minimize(self.db_loss, var_list=self.discriminatorB.variables)

    def train(self, opt, DB_A, DB_B):
        self.writer = tf.summary.FileWriter(opt.log_dir, self.sess.graph)
        disp_step_train = ceil(opt.nStep_train / opt.disp_div_N)

        latest_ckpt = tf.train.latest_checkpoint(opt.ckpt_dir)
        if latest_ckpt == None:
            print('Start initially!')
            self.sess.run(tf.global_variables_initializer())
            epoch_start = 0
        else:
            print('Start from saved model - ' + latest_ckpt)
            self.saver.restore(self.sess, latest_ckpt)
            epoch_start = myNumExt(latest_ckpt)

        total_st = time.time()

        for epoch in range(epoch_start, opt.nEpoch):
            epoch_st = time.time()
            disp_cnt = 1
            DB_A.shuffle(seed=777)
            DB_B.shuffle(seed=888)
            lr = opt.lr if epoch < opt.decay_epoch else opt.lr * (opt.nEpoch - epoch) / (opt.nEpoch - opt.decay_epoch)

            sum_g_loss_train = 0.0
            sum_d_loss_train = 0.0

            out_argG = [self.fake_A, self.fake_B, self.ga2b_optim, self.gb2a_optim, self.g_loss]
            out_argmG = [self.fake_A, self.fake_B, self.ga2b_optim, self.gb2a_optim, self.g_loss, self.g_sum,
                         self.ssos_sum]
            out_argD = [self.da_optim, self.db_optim, self.d_loss]
            out_argmD = [self.da_optim, self.db_optim, self.d_loss, self.d_sum, self.lr_sum]

            for step in range(opt.nStep_train):
                step_st = time.time()
                real_A, scale_A = DB_A.getBatch(step)
                real_B, scale_B = DB_B.getBatch(step)

                feed_dictG = {self.real_A: real_A, self.real_B: real_B, self.scale_A: scale_A, self.scale_B: scale_B,
                              self.lr: lr}

                if step % disp_step_train == 0:
                    fake_A, fake_B, _, _, g_loss_train, g_summary, ssos_summary = \
                        self.sess.run(out_argmG, feed_dict=feed_dictG)
                    self.writer.add_summary(g_summary, epoch * opt.disp_div_N + disp_cnt)
                    self.writer.add_summary(ssos_summary, epoch * opt.disp_div_N + disp_cnt)
                else:
                    fake_A, fake_B, _, _, g_loss_train = self.sess.run(out_argG, feed_dict=feed_dictG)

                [fake_A, fake_B] = self.pool([fake_A, fake_B])

                feed_dictD = {self.real_A: real_A, self.real_B: real_B, self.fake_A_sample: fake_A,
                              self.fake_B_sample: fake_B, self.lr: lr}

                for n in range(self.n_critic - 1):
                    _, _, d_loss_train = self.sess.run(out_argD, feed_dict=feed_dictD)
                    sum_d_loss_train += d_loss_train

                if step % disp_step_train == 0:
                    _, _, d_loss_train, d_summary, lr_summary = self.sess.run(out_argmD, feed_dict=feed_dictD)
                    self.writer.add_summary(d_summary, epoch * opt.disp_div_N + disp_cnt)
                    self.writer.add_summary(lr_summary, epoch * opt.disp_div_N + disp_cnt)
                    disp_cnt += 1
                else:
                    _, _, d_loss_train = self.sess.run(out_argD, feed_dict=feed_dictD)

                sum_g_loss_train += g_loss_train
                sum_d_loss_train += d_loss_train

                step_et = time.time()
                print('Epoch: [%3d] [%4d/%4d] g_loss: %2.4f, d_loss: %2.4f, time: %4.4f sec' %
                      (epoch + 1, step + 1, opt.nStep_train, g_loss_train, d_loss_train, step_et - step_st))
            epoch_et = time.time()
            print('Epoch: [%3d] g_loss: %2.4f, d_loss: %2.4f, time: %4.4f sec' %
                  (epoch + 1, sum_g_loss_train / opt.nStep_train, sum_d_loss_train / (opt.nStep_train * opt.n_critic),
                   epoch_et - epoch_st))

            if (epoch + 1) % 1 == 0:
                self.saver.save(self.sess, os.path.join(opt.ckpt_dir, 'model.ckpt'), global_step=epoch + 1)

        total_et = time.time()
        print('Total time elapsed: %4.4f sec' % (total_et - total_st))

    def test(self, opt, DB_B):
        latest_ckpt = tf.train.latest_checkpoint(opt.ckpt_dir)
        self.saver.restore(self.sess, latest_ckpt)

        out_arg = self.test_fake_img_A
        if not os.path.isdir(opt.savepath + '/' + opt.name + '/' + 'test/'):
            os.makedirs(opt.savepath + '/' + opt.name + '/' + 'test/')

        print('Start test')
        total_st = time.time()

        for step in range(opt.nStep_test):
            step_st = time.time()
            test_real_B, scale_B = DB_B.getBatch(step)

            feed_dict = {self.test_real_B: test_real_B, self.scale_B: scale_B}
            test_fake_A = self.sess.run(out_arg, feed_dict=feed_dict)

            subNum_B = DB_B.flist[step].split('_')[0]

            pre_str_B = opt.savepath + '/' + opt.name + '/' + 'test/' + subNum_B

            fake_A = {'fake': test_fake_A}

            sio.savemat(pre_str_B + '_th_fake_full.mat', fake_A)

            step_et = time.time()
            print('[%4d/%4d] time: %4.4f sec' % (step + 1, opt.nStep_test, step_et - step_st))

        total_et = time.time()
        print('Total time elapsed: %4.4f sec' % (total_et - total_st))


class CycleGAN_AccMRI_fft:
    def __init__(self, sess, opt):
        GPU = '/device:GPU:' + str(opt.gpu_ids[0])
        print(GPU)

        self.GPU = GPU
        self.sess = sess
        self.nX = opt.nX
        self.nC = opt.nC
        self.cyc_lambda = opt.cyc_lambda
        self.iden_lambda = opt.iden_lambda
        self.GP_lambda = opt.GP_lambda
        self.beta1 = opt.beta1
        self.beta2 = opt.beta2
        self.opt = opt
        self.generator = opt.generator
        self.discriminator = opt.discriminator
        self.use_identity = opt.use_identity
        self.use_wgan = opt.use_wgan
        self.n_critic = opt.n_critic
        self.DSrate = opt.DSrate

        if opt.use_lsgan:
            self.criterionGAN = mse_criterion
            self.use_sigmoid = False
        elif opt.use_wgan:
            self.use_sigmoid = False
        else:
            self.criterionGAN = sce_criterion
            self.use_sigmoid = True

        self._build_model()
        self.saver = tf.train.Saver()
        self.pool = ImagePool(opt.pool_size)

    def _build_model(self):
        with tf.device(self.GPU):
            self.generatorA2B = FFTGenerator(self.opt, name='generatorA2B')
            if self.generator == 'unet':
                self.generatorB2A = UnetGenerator(self.opt, name='generatorB2A')
            elif self.generator == 'unet_residual':
                self.generatorB2A = UnetGenerator(self.opt, res=True, name='generatorB2A')
            else:
                st()

            if self.discriminator == 'basic':
                self.discriminatorA = Discriminator(self.opt, name='discriminatorA', use_sigmoid=self.use_sigmoid)
            else:
                st()

            self.real_A = tf.placeholder(dtype, [1, None, self.nX, self.nC * 2], name='real_A')
            self.real_B = tf.placeholder(dtype, [1, None, self.nX, self.nC * 2], name='real_B')
            self.mask_A = tf.placeholder(dtype, [1, None, self.nX, self.nC], name='mask_A')
            self.mask_B = tf.placeholder(dtype, [1, None, self.nX, self.nC], name='mask_B')

            self.fake_B = self.generatorA2B(self.real_A, self.mask_A)
            self.fake_A = self.generatorB2A(self.real_B)
            self.recon_B = self.generatorA2B(self.fake_A, self.mask_B)
            self.recon_A = self.generatorB2A(self.fake_B)

            self.iden_A = self.generatorB2A(self.real_A)

            self.DA_fake = self.discriminatorA(self.fake_A)

            if self.use_wgan:
                self.gan_loss = -tf.reduce_mean(self.DA_fake)
                self.cycle_loss = mae_criterion(self.real_A, self.recon_A) + mae_criterion(self.real_B, self.recon_B)
                self.g_loss = self.gan_loss + self.cyc_lambda * self.cycle_loss
            else:
                self.gan_loss = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake))
                self.cycle_loss = mae_criterion(self.real_A, self.recon_A) + mae_criterion(self.real_B, self.recon_B)
                self.g_loss = self.gan_loss + self.cyc_lambda * self.cycle_loss

            if self.use_identity:
                self.iden_loss = mae_criterion(self.real_A, self.iden_A)
                self.g_loss = self.g_loss + self.iden_lambda * self.iden_loss

            self.fake_A_sample = tf.placeholder(dtype, [1, None, self.nX, self.nC * 2], name='fake_A_sample')
            self.DA_real = self.discriminatorA(self.real_A)
            self.DA_fake_sample = self.discriminatorA(self.fake_A_sample)

            if self.use_wgan:
                self.d_loss_real = -tf.reduce_mean(self.DA_real)
                self.d_loss_fake = tf.reduce_mean(self.DA_fake_sample)
                self.d_loss_GP = gradient_penalty(self.fake_A_sample, self.real_A, 1, self.discriminatorA)
                self.d_loss = (self.d_loss_real + self.d_loss_fake + self.GP_lambda * self.d_loss_GP) / 2
            else:
                self.d_loss_real = self.criterionGAN(self.DA_real, tf.ones_like(self.DA_real))
                self.d_loss_fake = self.criterionGAN(self.DA_fake_sample, tf.zeros_like(self.DA_fake_sample))
                self.d_loss = (self.d_loss_real + self.d_loss_fake) / 2

            self.gan_loss_sum = tf.summary.scalar('generator/gan_loss', self.gan_loss)
            self.cycle_loss_sum = tf.summary.scalar('generator/cycle_loss', self.cycle_loss)
            self.g_loss_sum = tf.summary.scalar('generator/g_loss', self.g_loss)

            if self.use_identity:
                self.iden_loss_sum = tf.summary.scalar('generator/identity_loss', self.iden_loss)
                self.g_sum = tf.summary.merge([self.gan_loss_sum, self.cycle_loss_sum,
                                               self.iden_loss_sum, self.g_loss_sum])
            else:
                self.g_sum = tf.summary.merge([self.gan_loss_sum, self.cycle_loss_sum, self.g_loss_sum])

            self.d_loss_real_sum = tf.summary.scalar('discriminator/da_loss_real', self.d_loss_real)
            self.d_loss_fake_sum = tf.summary.scalar('discriminator/da_loss_fake', self.d_loss_fake)
            self.d_loss_sum = tf.summary.scalar('discriminator/da_loss', self.d_loss)

            if self.use_wgan:
                self.GP_sum = tf.summary.scalar('discriminator/GP_loss', self.d_loss_GP)
                self.d_sum = tf.summary.merge([self.d_loss_real_sum, self.d_loss_fake_sum,
                                               self.GP_sum, self.d_loss_sum])
            else:
                self.d_sum = tf.summary.merge([self.d_loss_real_sum, self.d_loss_fake_sum, self.d_loss_sum])

            self.scale_A = tf.placeholder(dtype, [1, None, self.nX, self.nC * 2], name='scale_tensor_A')
            self.scale_B = tf.placeholder(dtype, [1, None, self.nX, self.nC * 2], name='scale_tensor_B')

            self.real_A_ssos_sum = tf.summary.image('ssos/real_full', tf_imgri2ssos(self.real_A * self.scale_A),
                                                    max_outputs=1)
            self.real_B_ssos_sum = tf.summary.image('ssos/real_down', tf_imgri2ssos(self.real_B * self.scale_B),
                                                    max_outputs=1)
            self.fake_A_ssos_sum = tf.summary.image('ssos/fake_full', tf_imgri2ssos(self.fake_A * self.scale_B),
                                                    max_outputs=1)
            self.fake_B_ssos_sum = tf.summary.image('ssos/fake_down', tf_imgri2ssos(self.fake_B * self.scale_A),
                                                    max_outputs=1)
            self.recon_A_ssos_sum = tf.summary.image('ssos/recon_full', tf_imgri2ssos(self.recon_A * self.scale_A),
                                                     max_outputs=1)
            self.recon_B_ssos_sum = tf.summary.image('ssos/recon_down', tf_imgri2ssos(self.recon_B * self.scale_B),
                                                     max_outputs=1)
            self.ssos_sum = tf.summary.merge([self.real_A_ssos_sum, self.real_B_ssos_sum,
                                              self.fake_A_ssos_sum, self.fake_B_ssos_sum,
                                              self.recon_A_ssos_sum, self.recon_B_ssos_sum])

            self.test_real_B = tf.placeholder(dtype, [1, None, self.nX, self.nC * 2], name='test_real_B')

            self.test_fake_A = self.generatorB2A(self.test_real_B)

            self.test_fake_img_A = tf.squeeze(tf_ri2comp(self.test_fake_A * self.scale_B))

            self.lr = tf.placeholder(dtype, None, name='learning_rate')
            self.lr_sum = tf.summary.scalar('learning_rate', self.lr)

            self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=self.beta1, beta2=self.beta2). \
                minimize(self.g_loss, var_list=self.generatorB2A.variables)
            self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=self.beta1, beta2=self.beta2). \
                minimize(self.d_loss, var_list=self.discriminatorA.variables)

    def train(self, opt, DB_A, DB_B):
        self.writer = tf.summary.FileWriter(opt.log_dir, self.sess.graph)
        disp_step_train = ceil(opt.nStep_train / opt.disp_div_N)

        latest_ckpt = tf.train.latest_checkpoint(opt.ckpt_dir)
        if latest_ckpt == None:
            print('Start initially!')
            self.sess.run(tf.global_variables_initializer())
            epoch_start = 0
        else:
            print('Start from saved model - ' + latest_ckpt)
            self.saver.restore(self.sess, latest_ckpt)
            epoch_start = myNumExt(latest_ckpt)

        total_st = time.time()

        for epoch in range(epoch_start, opt.nEpoch):
            epoch_st = time.time()
            disp_cnt = 1
            DB_A.shuffle(seed=777)
            DB_B.shuffle(seed=888)
            lr = opt.lr if epoch < opt.decay_epoch else opt.lr * (opt.nEpoch - epoch) / (opt.nEpoch - opt.decay_epoch)

            sum_g_loss_train = 0.0
            sum_d_loss_train = 0.0

            out_argG = [self.fake_A, self.fake_B, self.g_optim, self.g_loss]
            out_argmG = [self.fake_A, self.fake_B, self.g_optim, self.g_loss, self.g_sum, self.ssos_sum]
            out_argD = [self.d_optim, self.d_loss]
            out_argmD = [self.d_optim, self.d_loss, self.d_sum, self.lr_sum]

            for step in range(opt.nStep_train):
                step_st = time.time()
                real_A, scale_A = DB_A.getBatch(step)
                real_B, scale_B, mask_B = DB_B.getBatch_fft(step)
                nY = np.shape(real_A)[1]
                mask_A = make_mask(nY, self.nX, self.nC, self.DSrate)

                feed_dictG = {self.real_A: real_A, self.real_B: real_B, self.mask_A: mask_A, self.mask_B: mask_B,
                              self.scale_A: scale_A, self.scale_B: scale_B, self.lr: lr}

                if step % disp_step_train == 0:
                    fake_A, fake_B, _, g_loss_train, g_summary, ssos_summary = \
                        self.sess.run(out_argmG, feed_dict=feed_dictG)
                    self.writer.add_summary(g_summary, epoch * opt.disp_div_N + disp_cnt)
                    self.writer.add_summary(ssos_summary, epoch * opt.disp_div_N + disp_cnt)
                else:
                    fake_A, fake_B, _, g_loss_train = self.sess.run(out_argG, feed_dict=feed_dictG)

                [fake_A, _] = self.pool([fake_A, fake_B])

                feed_dictD = {self.real_A: real_A, self.fake_A_sample: fake_A, self.lr: lr}

                for n in range(self.n_critic - 1):
                    _, d_loss_train = self.sess.run(out_argD, feed_dict=feed_dictD)
                    sum_d_loss_train += d_loss_train

                if step % disp_step_train == 0:
                    _, d_loss_train, d_summary, lr_summary = self.sess.run(out_argmD, feed_dict=feed_dictD)
                    self.writer.add_summary(d_summary, epoch * opt.disp_div_N + disp_cnt)
                    self.writer.add_summary(lr_summary, epoch * opt.disp_div_N + disp_cnt)
                    disp_cnt += 1
                else:
                    _, d_loss_train = self.sess.run(out_argD, feed_dict=feed_dictD)

                sum_g_loss_train += g_loss_train
                sum_d_loss_train += d_loss_train

                step_et = time.time()
                print('Epoch: [%3d] [%4d/%4d] g_loss: %2.4f, d_loss: %2.4f, time: %4.4f sec' %
                      (epoch + 1, step + 1, opt.nStep_train, g_loss_train, d_loss_train, step_et - step_st))
            epoch_et = time.time()
            print('Epoch: [%3d] g_loss: %2.4f, d_loss: %2.4f, time: %4.4f sec' %
                  (epoch + 1, sum_g_loss_train / opt.nStep_train, sum_d_loss_train / (opt.nStep_train * opt.n_critic),
                   epoch_et - epoch_st))

            if (epoch + 1) % 1 == 0:
                self.saver.save(self.sess, os.path.join(opt.ckpt_dir, 'model.ckpt'), global_step=epoch + 1)

        total_et = time.time()
        print('Total time elapsed: %4.4f sec' % (total_et - total_st))

    def test(self, opt, DB_B):
        latest_ckpt = tf.train.latest_checkpoint(opt.ckpt_dir)
        self.saver.restore(self.sess, latest_ckpt)

        out_arg = self.test_fake_img_A

        if not os.path.isdir(opt.savepath + '/' + opt.name + '/' + 'test/'):
            os.makedirs(opt.savepath + '/' + opt.name + '/' + 'test/')

        print('Start test')
        total_st = time.time()

        for step in range(opt.nStep_test):
            step_st = time.time()
            test_real_B, scale_B = DB_B.getBatch(step)

            feed_dict = {self.test_real_B: test_real_B, self.scale_B: scale_B}
            test_fake_A = self.sess.run(out_arg, feed_dict=feed_dict)

            subNum_B = DB_B.flist[step].split('_')[0]

            pre_str_B = opt.savepath + '/' + opt.name + '/' + 'test/' + subNum_B

            fake_A = {'fake': test_fake_A}

            sio.savemat(pre_str_B + '_th_fake_full.mat', fake_A)

            step_et = time.time()
            print('[%4d/%4d] time: %4.4f sec' % (step + 1, opt.nStep_test, step_et - step_st))

        total_et = time.time()
        print('Total time elapsed: %4.4f sec' % (total_et - total_st))


class Supervised_UNet:
    def __init__(self, sess, opt):
        GPU = '/device:GPU:' + str(opt.gpu_ids[0])
        print(GPU)

        self.GPU = GPU
        self.sess = sess
        self.nX = opt.nX
        self.nC = opt.nC
        self.beta1 = opt.beta1
        self.beta2 = opt.beta2
        self.opt = opt
        self.generator = opt.generator

        self._build_model()
        self.saver = tf.train.Saver()

    def _build_model(self):
        with tf.device(self.GPU):
            self.generator = UnetGenerator(self.opt, name='generator')

            self.target = tf.placeholder(dtype, [1, None, self.nX, self.nC * 2], name='real_A')
            self.input = tf.placeholder(dtype, [1, None, self.nX, self.nC * 2], name='real_B')

            self.output = self.generator(self.input)

            self.scale_target = tf.placeholder(dtype, [1, None, self.nX, self.nC * 2], name='scale_tensor_A')
            self.scale_input = tf.placeholder(dtype, [1, None, self.nX, self.nC * 2], name='scale_tensor_B')

            self.loss = mse_criterion(self.output, self.target * self.scale_target / self.scale_input)

            self.loss_sum = tf.summary.scalar('loss', self.loss)

            self.target_ssos_sum = tf.summary.image('ssos/target', tf_imgri2ssos(self.target * self.scale_target),
                                                    max_outputs=1)
            self.input_ssos_sum = tf.summary.image('ssos/input', tf_imgri2ssos(self.input * self.scale_input),
                                                   max_outputs=1)
            self.output_ssos_sum = tf.summary.image('ssos/output', tf_imgri2ssos(self.output * self.scale_input),
                                                    max_outputs=1)
            self.ssos_sum = tf.summary.merge([self.target_ssos_sum, self.input_ssos_sum, self.output_ssos_sum])

            self.test_input = tf.placeholder(dtype, [1, None, self.nX, self.nC * 2], name='test_real_B')

            self.test_output = self.generator(self.test_input)

            self.test_output_img = tf.squeeze(tf_ri2comp(self.test_output * self.scale_input))

            self.lr = tf.placeholder(dtype, None, name='learning_rate')
            self.lr_sum = tf.summary.scalar('learning_rate', self.lr)

            self.optim = tf.train.AdamOptimizer(self.lr, beta1=self.beta1, beta2=self.beta2). \
                minimize(self.loss, var_list=self.generator.variables)

    def train(self, opt, DB_A, DB_B):
        self.writer = tf.summary.FileWriter(opt.log_dir, self.sess.graph)
        disp_step_train = ceil(opt.nStep_train / opt.disp_div_N)

        latest_ckpt = tf.train.latest_checkpoint(opt.ckpt_dir)
        if latest_ckpt == None:
            print('Start initially!')
            self.sess.run(tf.global_variables_initializer())
            epoch_start = 0
        else:
            print('Start from saved model - ' + latest_ckpt)
            self.saver.restore(self.sess, latest_ckpt)
            epoch_start = myNumExt(latest_ckpt)

        total_st = time.time()

        for epoch in range(epoch_start, opt.nEpoch):
            epoch_st = time.time()
            disp_cnt = 1
            DB_A.shuffle(seed=777)
            DB_B.shuffle(seed=777)
            lr = opt.lr if epoch < opt.decay_epoch else opt.lr * (opt.nEpoch - epoch) / (opt.nEpoch - opt.decay_epoch)

            sum_loss_train = 0.0

            out_arg = [self.optim, self.loss]
            out_argm = [self.optim, self.loss, self.loss_sum, self.ssos_sum, self.lr_sum]

            for step in range(opt.nStep_train):
                step_st = time.time()
                target, scale_target = DB_A.getBatch(step)
                input, scale_input = DB_B.getBatch(step)

                feed_dict = {self.target: target, self.input: input,
                             self.scale_target: scale_target, self.scale_input: scale_input, self.lr: lr}

                if step % disp_step_train == 0:
                    _, loss_train, summary, ssos_summary, lr_summary = self.sess.run(out_argm, feed_dict=feed_dict)
                    self.writer.add_summary(summary, epoch * opt.disp_div_N + disp_cnt)
                    self.writer.add_summary(ssos_summary, epoch * opt.disp_div_N + disp_cnt)
                    self.writer.add_summary(lr_summary, epoch * opt.disp_div_N + disp_cnt)
                    disp_cnt += 1
                else:
                    _, loss_train = self.sess.run(out_arg, feed_dict=feed_dict)

                sum_loss_train += loss_train

                step_et = time.time()
                print('Epoch: [%3d] [%4d/%4d] loss: %2.4f e-2, time: %4.4f sec' %
                      (epoch + 1, step + 1, opt.nStep_train, loss_train * 1e2, step_et - step_st))
            epoch_et = time.time()
            print('Epoch: [%3d] loss: %2.4f e-2, time: %4.4f sec' %
                  (epoch + 1, sum_loss_train * 1e2 / opt.nStep_train, epoch_et - epoch_st))

            if (epoch + 1) % 1 == 0:
                self.saver.save(self.sess, os.path.join(opt.ckpt_dir, 'model.ckpt'), global_step=epoch + 1)

        total_et = time.time()
        print('Total time elapsed: %4.4f sec' % (total_et - total_st))

    def test(self, opt, DB_B):
        latest_ckpt = tf.train.latest_checkpoint(opt.ckpt_dir)
        self.saver.restore(self.sess, latest_ckpt)

        out_arg = self.test_output_img

        if not os.path.isdir(opt.savepath + '/' + opt.name + '/' + 'test/'):
            os.makedirs(opt.savepath + '/' + opt.name + '/' + 'test/')

        print('Start test')
        total_st = time.time()

        for step in range(opt.nStep_test):
            step_st = time.time()
            test_input, scale_input = DB_B.getBatch(step)

            feed_dict = {self.test_input: test_input, self.scale_input: scale_input}
            test_output = self.sess.run(out_arg, feed_dict=feed_dict)

            subNum_B = DB_B.flist[step].split('_')[0]

            pre_str_B = opt.savepath + '/' + opt.name + '/' + 'test/' + subNum_B

            fake_A = {'fake': test_output}

            sio.savemat(pre_str_B + '_th_fake_full.mat', fake_A)

            step_et = time.time()
            print('[%4d/%4d] time: %4.4f sec' % (step + 1, opt.nStep_test, step_et - step_st))

        total_et = time.time()
        print('Total time elapsed: %4.4f sec' % (total_et - total_st))
