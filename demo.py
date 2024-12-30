#!/usr/bin/env python

import os
import sys
from pprint import pprint
from PIL import Image

# Pytorch requires blocking launch for proper working
if sys.platform == 'win32':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import numpy as np
from scipy import io

import torch
import torch.nn

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

sys.path.append('modules')
from modules import utils
from modules import motion
from modules import dataset
from modules import thermal

if __name__ == '__main__':
    imname = 'test_rgb2'  # Name of the test file name
    imname_thermal = 'test_thermal2'
    camera = 'sim'  # 'sim', 'boson' or 'lepton'
    scale_sr = 16  # 1 for denoising/NUC, 2, 3, .. for SR
    nimg = 20  # Number of input images

    method = 'dip'  # 'cvx' for Hardie et al., 'dip' for DeepIR

    # Load config file -- 
    config = dataset.load_config('configs/%s_%s.ini' % (method, camera))
    config['batch_size'] = nimg
    config['num_workers'] = (0 if sys.platform == 'win32' else 4)
    config['lambda_prior'] *= (scale_sr / nimg)

    # Load data
    if not config['real']:
        # This is simulated data
        im = utils.get_img(imname, 1)
        print(im.shape)
        minval = 0
        maxval = 1

    else:
        # This is real data
        im, minval, maxval = utils.get_real_im(imname, camera)
    im_t = utils.get_img_t(imname_thermal, 1)  # send in thermal images.
    print(im_t.shape)
    # Get data for SR -- this will also get an initial estimate for registration
    im, imstack, imstack_t, ecc_mats, ecc_t_mats = motion.get_SR_data(im, im_t, scale_sr, nimg, config) # added im_t
    ecc_mats[:, :, 2] *= scale_sr
    ecc_t_mats[:, :, 2] *= scale_sr
    H, W = im.shape
    Ht, Wt = im_t.shape

    # Load LPIPs function
    config['gt'] = im

    # Now run denoising
    # if method == 'cvx':
    #     im_dip, profile_dip = thermal.interp_convex(imstack.astype(np.float32),
    #                                             ecc_mats.astype(np.float32),
    #                                             (H, W), config)
    # else:
    #     im_dip, profile_dip = thermal.interp_DIP(imstack.astype(np.float32),
    #                                             ecc_mats.astype(np.float32),
    #                                             (H, W), config)
    # Save data
    mdict = {'imstack': imstack,
             'ecc_mats': ecc_mats}
    mdict_t = {'imstack_t': imstack_t,
               'ecc_t_mats': ecc_t_mats}
    io.savemat('%s_%s_%s_%dx_%d.mat' % (imname, camera, method,
                                        scale_sr, nimg), mdict)
    io.savemat('%s_%s_%s_%dx_%d.mat' % (imname_thermal, camera, method,
                                        scale_sr, nimg), mdict_t)



