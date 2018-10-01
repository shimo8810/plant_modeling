import random
import argparse
from pathlib import Path
from tqdm import tqdm
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
    parser.add_argument('--dec', '-d', type=str, default='prob_dec_model.npz',
        help='decoder model')
    parser.add_argument('--enc', '-e', type=str, default='prob_enc_model.npz',
        help='encoder model')
    parser.add_argument('--numval', '-n', type=int, default=10,
        help='encoder model')
    args = parser.parse_args()

    enc = Encoder(in_ch=3)
    dec = Decoder(out_ch=1)

    chainer.serializers.load_npz(str(MODEL_PATH.joinpath(args.dec)), dec)
    chainer.serializers.load_npz(str(MODEL_PATH.joinpath(args.enc)), enc)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        enc.to_gpu()  # Copy the model to the GPU
        dec.to_gpu()
    xp = enc.xp

    branch_images = np.zeros((12, 256, 256), dtype=np.uint8)
    plant_images = np.zeros((12, 256, 256, 3), dtype=np.uint8)
    p2b_images = np.zeros((12, 256, 256), dtype=np.uint8)

    for i in tqdm(range(1,12)):
        branch_path = DATASET_PATH.joinpath('branch', str(i))
        plant_path = DATASET_PATH.joinpath('plant', str(i))
        name = random.choice([_ for _ in branch_path.glob('*.png')]).name
        branch_image_path = branch_path.joinpath(name)
        plant_image_path = plant_path.joinpath(name)

        # open image
        branch_image = np.asarray(Image.open(branch_image_path).convert('L'))
        branch_images[i-1,:] = branch_image
        plant_image = np.asarray(Image.open(plant_image_path).convert('RGB'))
        plant_images[i-1,:] = plant_image

        plant_image = xp.array(plant_image).astype("f").transpose(2, 0, 1) / 128.0-1.0
        plant_image = plant_image.reshape(1, *plant_image.shape)

        probs = np.zeros((256, 256))
        for _ in range(args.numval):
            with chainer.no_backprop_mode():
                p2b_image = np.asarray(dec(enc(plant_image)).data.get()).reshape(256, 256)
            p2b_image = (p2b_image + 1.0) / 2.0
            probs = probs + p2b_image
        probs /= args.numval
        p2b_image = np.asarray(np.clip(probs * 255, 0.0, 255.0), dtype=np.uint8)

        p2b_images[i-1, :] = p2b_image

    Image.fromarray(branch_images.reshape(3, 4, 256, 256).transpose(0, 2, 1, 3).reshape(3*256, 4*256))\
        .save(str(RESULT_PATH.joinpath('prob_branch_image.png')))
    Image.fromarray(plant_images.reshape(3, 4, 256, 256, 3).transpose(0, 2, 1, 3, 4).reshape(3*256, 4*256, 3))\
        .save(str(RESULT_PATH.joinpath('prob_plant_image.png')))
    Image.fromarray(p2b_images.reshape(3, 4, 256, 256).transpose(0, 2, 1, 3).reshape(3*256, 4*256))\
        .save(str(RESULT_PATH.joinpath('prob_p2b_image.png')))

if __name__ == '__main__':
    main()
