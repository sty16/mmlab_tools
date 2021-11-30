import os
import shutil
from sklearn.model_selection import train_test_split
import glob


class Data2Imagenet:
    def __init__(self, saved_imagenet_path):
        self.ann = []
        self.saved_imagenet_path = saved_imagenet_path

    def to_imagenet(self, root):
        folders = [
            d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
        ]
        folders.sort()
        folder_to_idx = {folders[i]: i for i in range(len(folders))}
        for folder_name in folders:
            _dir = os.path.join(root, folder_name)
            fns = glob.glob(_dir + "/*.jpg")
            label = folder_to_idx[folder_name]
            train_fns, val_fns = train_test_split(fns, test_size=0.20)
            for fn in train_fns:
                if not os.path.exists(os.path.join(saved_imagenet_path, 'train', folder_name)):
                    os.makedirs(os.path.join(saved_imagenet_path, 'train', folder_name))
                shutil.copy(fn, f'{self.saved_imagenet_path}/train/{folder_name}/')
            for fn in val_fns:
                self.ann.append((fn, label))
                shutil.copy(fn, f'{self.saved_imagenet_path}/val/')
        self.save_imagenet_txt()

    def save_imagenet_txt(self):
        val_file = os.path.join(self.saved_imagenet_path, 'meta', 'val.txt')
        with open(val_file, 'w') as f:
            for fn, label in self.ann:
                f.write(f'{fn} {label}\n')


if __name__ == '__main__':
    data_path = '/Data/cell/group_1_small_classify'
    saved_imagenet_path = '/Data/cell_imagenet/imagenet'
    if not os.path.exists(os.path.join(saved_imagenet_path, 'train')):
        os.makedirs(os.path.join(saved_imagenet_path, 'train'))
    if not os.path.exists(os.path.join(saved_imagenet_path, 'val')):
        os.makedirs(os.path.join(saved_imagenet_path, 'val'))
    if not os.path.exists(os.path.join(saved_imagenet_path, 'meta')):
        os.makedirs(os.path.join(saved_imagenet_path, 'meta'))

    data2imagenet = Data2Imagenet(saved_imagenet_path)
    data2imagenet.to_imagenet(data_path)







