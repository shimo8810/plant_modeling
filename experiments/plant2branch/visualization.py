import random
import argparse
from pathlib import Path

from PIL import Image
import numpy as np
import chainer
from net import Decoder, Encoder

# PATH 関連
FILE_PATH = Path(__file__).resolve().parent
ROOT_PATH = FILE_PATH.parent.parent
RESULT_PATH = ROOT_PATH.joinpath('results/plant2branch')
DATASET_PATH = ROOT_PATH.joinpath('datasets/plant2branch')
MODEL_PATH = ROOT_PATH.joinpath('models/plant2branch')

def main():
    parser = argparse.ArgumentParser(
        description='chainer implementation of pix2pix')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
        help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    enc = Encoder(in_ch=3)
    dec = Decoder(out_ch=1)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        enc.to_gpu()  # Copy the model to the GPU
        dec.to_gpu()
    xp = enc.xp

    branch_images = np.zeros((12, 256, 256))
    plant_images = np.zeros((12, 256, 256, 3))
    p2b_images = np.zeros((12, 256, 256))
    for i in range(1,12):
        branch_path = DATASET_PATH.joinpath('branch', str(i))
        plant_path = DATASET_PATH.joinpath('plant', str(i))
        name = random.choice([_ for _ in branch_path.glob('*.png')]).name
        branch_image_path = branch_path.joinpath(name)
        plant_image_path = plant_path.joinpath(name)

        # open image
        branch_image = np.asarray(Image.open(branch_image_path).convert('L'))
        branch_images[i,:] = branch_image
        plant_image = np.asarray(Image.open(plant_image_path).convert('RGB'))
        plant_images[i,:] = plant_image

        plant_image = xp.array(plant_image).astype("f").transpose(2, 0, 1) / 128.0-1.0
        plant_image = plant_image.reshape(1, *plant_image.shape)
        prob = np.ones((1, 1))
        with chainer.no_backprop_mode():
            pass
            # for j in range(1):
            #     p2b_image = np.clip(dec(enc(plant_image)).data.get()[0,:], -1.0, 1.0) * 128 + 128
            #     print(p2b_image.shape, p2b_image.max(), p2b_image.min())
            #     # p2b_image = np.asarray(np.clip(p2b_image * 128 + 128, 0.0, 255.0), dtype=np.uint8).reshape(256,256)
            #     # Image.fromarray(p2b_image).save('{}.png'.format(j))
        break




if __name__ == '__main__':
    main()
