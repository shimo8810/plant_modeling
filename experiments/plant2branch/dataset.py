from pathlib import Path
from PIL import Image
from chainer.dataset import dataset_mixin
import numpy as np

# PATH 関連
FILE_PATH = Path().resolve()
ROOT_PATH = FILE_PATH.parent.parent
DATASET_PATH = ROOT_PATH.joinpath('datasets')

class P2BDataset(dataset_mixin.DatasetMixin):
    def __init__(self, data_range=(1, 11)):
        data_path = DATASET_PATH.joinpath('plant2branch')
        print('loading dataset ...')
        print('data path: {}'.format(data_path))
        self.data = []
        #各品種に対して
        for i in range(data_range[0], data_range[1]):
            branch_path = data_path.joinpath('branch', str(i))
            plant_path = data_path.joinpath('plant', str(i))
            # print(branch_path)
            # print(plant_path)
            # 各画像に対して
            for branch_image_path in branch_path.glob('*.png'):
                plant_image_path = plant_path.joinpath(branch_image_path.name)
                # print(branch_image_path)
                # print(plant_image_path)
                branch_image = np.asarray(Image.open(branch_image_path).convert('L')).astype("f").reshape(1, 256, 256) / 128.0-1.0
                plant_image = np.asarray(Image.open(plant_image_path).convert('RGB')).astype("f").transpose(2, 0, 1) / 128.0-1.0
                # print(plant_image.shape, plant_image.dtype)
                # print(branch_image.shape, branch_image.dtype)

                self.data.append((plant_image, branch_image))
        print('load dataset done!')

    def __len__(self):
        return len(self.data)

    def get_example(self, i):
        """
        return (plant image, branch image)
        """
        return self.data[i]

if __name__ == '__main__':
    # test
    dataset = P2BDataset()
    print(len(dataset))
    dataset = P2BDataset(data_range=(11, 12))
    print(len(dataset))
