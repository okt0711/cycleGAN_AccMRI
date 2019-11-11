from os import listdir
from os.path import join, isfile
import random
from scipy import io as sio
import numpy as np

dtype = np.float32


class DB:
    def __init__(self, opt, phase, domain):
        super(DB, self).__init__()
        random.seed(777)
        if domain == 'A':
            self.dataroot = opt.dataroot_A
        else:
            self.dataroot = opt.dataroot_B
        self.root = join(self.dataroot, phase)
        self.flist = []

        list_fname = join(self.dataroot, 'DBset_' + phase + '.npy')
        if isfile(list_fname):
            print(list_fname + ' exists. Now loading...')
            self.flist = np.load(list_fname)
        else:
            print('Now generating...' + list_fname)
            flist = []
            for aImg in sorted(listdir(self.root)):
                flist.append(aImg)
            np.save(list_fname, flist)
            self.flist = flist

        if opt.small_DB and phase == 'train':
            self.flist = self.flist[:1000]

        self.nC = opt.nC
        self.nCh_in = opt.nC * 2
        self.nY = opt.nY
        self.nX = opt.nX
        self.len = len(self.flist)

    def getBatch(self, step):
        batch = self.flist[step]

        aInput_img, vec_max = self.read_mat_std(join(self.root, batch), 'std_ri')

        nY = aInput_img.shape[0]
        sz = [1, nY, self.nX, self.nCh_in]

        input_img_ri = np.empty(sz, dtype=dtype)
        scale_std = np.empty([1, self.nCh_in], dtype=dtype)

        scale_std[0, 0:self.nC] = vec_max[0, :]
        scale_std[0, self.nC:] = vec_max[0, :]
        scale_std = np.tile(scale_std[:, np.newaxis, np.newaxis, :], [1, nY, self.nX, 1])

        input_img_ri[0, :, :, :] = aInput_img

        return input_img_ri, scale_std

    def getBatch_fft(self, step):
        batch = self.flist[step]
        # print(self.flist[step])

        aInput_img, vec_max, mask = self.read_mat_std_mask(join(self.root, batch), 'std_ri')

        nY = aInput_img.shape[0]
        sz = [1, nY, self.nX, self.nCh_in]

        input_img_ri = np.empty(sz, dtype=dtype)
        scale_std = np.empty([1, self.nCh_in], dtype=dtype)
        masks = np.empty([1, nY, self.nX, self.nC], dtype=dtype)

        scale_std[0, 0:self.nC] = vec_max[0, :]
        scale_std[0, self.nC:] = vec_max[0, :]
        scale_std = np.tile(scale_std[:, np.newaxis, np.newaxis, :], [1, nY, self.nX, 1])

        mask = np.fft.fftshift(mask)
        masks[0, :, :, :] = np.tile(mask[:, :, np.newaxis], [1, 1, self.nC])

        input_img_ri[0, :, :, :] = aInput_img

        return input_img_ri, scale_std, masks

    def shuffle(self, seed=0):
        random.seed(seed)
        random.shuffle(self.flist)

    @staticmethod
    def read_mat_std(filename, var_name='std_kri'):
        mat = sio.loadmat(filename)
        return mat[var_name], mat['vec_std']

    @staticmethod
    def read_mat_std_mask(filename, var_name='std_kri'):
        mat = sio.loadmat(filename)
        return mat[var_name], mat['vec_std'], mat['mask4_1D']
