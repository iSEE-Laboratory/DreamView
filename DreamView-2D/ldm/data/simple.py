# Copyright (c) Alibaba, Inc. and its affiliates.
# @author:  Drinky Yan
# @contact: yanjk3@mail2.sysu.edu.cn
import webdataset as wds
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from einops import rearrange
import pytorch_lightning as pl
import random
import matplotlib.pyplot as plt
import os.path as osp
import pandas as pd
from .sampler import Combined2DAnd3DSampler
import pickle as pkl


class Combined2DAnd3DModuleFromConfig(pl.LightningDataModule):
    def __init__(self,
                 root_dir_2d,
                 root_dir_3d,
                 batch_size,
                 total_view,
                 num_workers):
        super().__init__()
        self.root_dir_2d = root_dir_2d
        self.root_dir_3d = root_dir_3d
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.total_view = total_view

        self.image_transforms = transforms.Compose([transforms.Resize((256, 256)),
                                                    transforms.ToTensor(),
                                                    transforms.Lambda(
                                                        lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])

    def train_dataloader(self):
        dataset = Combined2DAnd3DData(root_dir_2d=self.root_dir_2d,
                                      root_dir_3d=self.root_dir_3d,
                                      total_view=self.total_view,
                                      image_transforms=self.image_transforms,
                                      validation=False)
        sampler = Combined2DAnd3DSampler(dataset, batch_size=self.batch_size)
        return wds.WebLoader(dataset,
                             batch_size=self.batch_size,
                             num_workers=self.num_workers,
                             shuffle=False,
                             sampler=sampler)

    def val_dataloader(self):
        dataset = Combined2DAnd3DData(root_dir_2d=self.root_dir_2d,
                                      root_dir_3d=self.root_dir_3d,
                                      total_view=self.total_view,
                                      image_transforms=self.image_transforms,
                                      validation=True)
        return wds.WebLoader(dataset,
                             batch_size=1,
                             num_workers=self.num_workers,
                             shuffle=False)


class Combined2DAnd3DData(Dataset):
    def __init__(self,
                 root_dir_2d,
                 root_dir_3d,
                 total_view,
                 image_transforms=None,
                 validation=False):
        self.root_dir_2d = root_dir_2d
        self.root_dir_3d = root_dir_3d
        self.validation = validation
        self.transforms = image_transforms

        # 3D Dataset
        self.total_view = total_view
        with open(osp.join(self.root_dir_3d, 'all_caption.pkl'), 'rb') as f:
            self.data_3d = pkl.load(f)

        keys_3d = list(self.data_3d.keys())
        if validation:
            self.keys_3d = keys_3d[int(len(keys_3d) * 0.99):]
        else:
            keys_3d = keys_3d[:int(len(keys_3d) * 0.99)]
            random.shuffle(keys_3d)
            self.keys_3d = keys_3d

        self.split_length = len(self.keys_3d)
        print('============= number of objects in 3D dataset %d =============' % len(self.keys_3d))

        # 2D Dataset
        self.data_2d = pd.read_csv(osp.join(self.root_dir_2d, 'pair.csv'), header=None)
        if validation:
            self.data_2d = self.data_2d[:1000]
        print('============= number of samples in 2D dataset %d =============' % len(self.data_2d))

        self.idx = {i: 26 - i if i < 26 else 58 - i for i in range(1, 33)}

    def __len__(self):
        return len(self.keys_3d) + len(self.data_2d)

    @staticmethod
    def load_im(path):
        img = plt.imread(path)
        if path.endswith('png'):
            img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        return img

    def __getitem__(self, index):
        data = {}

        # return 3D data
        if index < self.split_length:
            view = random.randint(0, self.total_view - 1)
            view_list = [((view + x) % 32) + 1 for x in [0, 8, 16, 24]]

            success = False
            while not success:
                obj_id = self.keys_3d[index]
                all_caption = self.data_3d[obj_id]  # 0 is global caption
                imgs, texts, g_texts, cameras = [], [], [], []

                for view in view_list:
                    img_path = osp.join(self.root_dir_3d, 'view_image', str(view), obj_id + '.png')
                    img = self.process_im(self.load_im(img_path))
                    imgs.append(img)

                    texts.append(all_caption[view] + ', 3D asset')
                    g_texts.append(all_caption[0] + ', 3D asset')

                    # 180 degree is the front view                
                    # camera_path = osp.join(self.root_dir_3d, 'view_camera', str(view), obj_id + '.npy')

                    # 90 degree is the front view
                    camera_path = osp.join(self.root_dir_3d, 'view_camera', str(self.idx[view]), obj_id + '.npy')

                    try:
                        camera = np.load(camera_path)
                    except ValueError:
                        index = (index + random.randint(1, self.split_length - 1)) % self.split_length
                        break
                    translation = camera[:3, 3]
                    translation = translation / (np.linalg.norm(translation, axis=0, keepdims=True) + 1e-8)
                    camera[:3, 3] = translation
                    cameras.append(camera)

                if len(cameras) == 4:
                    success = True

            data['imgs'] = imgs
            data['texts'] = texts
            data['g_texts'] = g_texts
            data['cameras'] = cameras

        # return 2D data
        else:
            img_path = osp.join(self.root_dir_2d, 'data', self.data_2d[0][index - self.split_length])
            img = self.process_im(self.load_im(img_path))
            data['img'] = img
            data['text'] = self.data_2d[1][index - self.split_length]

        return data

    def process_im(self, im):
        im = im.convert("RGB")
        if self.transforms:
            return self.transforms(im)
        else:
            return im
