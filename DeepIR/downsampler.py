import os
import sys
from pprint import pprint

import numpy as np
from scipy import io
from PIL import Image
from modules import utils
from modules import motion
from modules import dataset
from modules import thermal

if __name__ == '__main__':
    imname = 'og_book'  # Name of the test file name
    camera = 'sim'  # 'sim', 'boson' or 'lepton'
    scale_sr = 8  # 1 for denoising/NUC, 2, 3, .. for SR
    nimg = 20  # Number of input images

    method = 'dip'  # 'cvx' for Hardie et al., 'dip' for DeepIR

    config = dataset.load_config('configs/%s_%s.ini' % (method, camera))
    config['batch_size'] = nimg
    config['num_workers'] = (0 if sys.platform == 'win32' else 4)
    config['lambda_prior'] *= (scale_sr / nimg)

    im = utils.get_img(imname, 1)
    minval = 0
    maxval = 1

    im, imstack, ecc_mats = motion.get_SR_data(im, scale_sr, nimg, config)
    ecc_mats[:, :, 2] *= scale_sr

    print(imstack.shape)

    io.savemat('stack.mat', {'stack': imstack})

mat_file = io.loadmat('/Users/giridharpeddi/OnChip/Compilations/thermal/DeepIR/stack.mat')

# Extract the image data (assuming it's stored as a variable named 'images')
image_data = mat_file['stack']

# Create a folder to save the PNG images
output_folder = '/Users/giridharpeddi/OnChip/Compilations/thermal/stich/results/low_rez_test'
os.makedirs(output_folder, exist_ok=True)

# Loop through the images and save them as PNG files
for i, image in enumerate(image_data):
    # Normalize the image data if necessary (adjust the scaling as needed)
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    image = image.astype(np.uint8)

    # Convert the NumPy array to a PIL Image
    pil_image = Image.fromarray(image)

    # Specify the output filename (customize as needed)
    image_filename = os.path.join(output_folder, f'image_{i:03d}.png')

    # Save the image as a PNG file
    pil_image.save(image_filename)

print(f"{len(image_data)} images saved as PNGs in {output_folder}")
