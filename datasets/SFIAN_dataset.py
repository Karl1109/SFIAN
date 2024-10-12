import os.path
import random
import cv2
import numpy as np
from PIL import Image
from datasets.base_dataset import BaseDataset
import torchvision.transforms as transforms
from datasets.image_folder import make_dataset
from datasets.utils import MaskToTensor, get_params, affine_transform


class SFIANDataset(BaseDataset):
    """A dataset class for crack dataset."""

    def __init__(self, opt):

        BaseDataset.__init__(self, opt)
        self.img_paths = make_dataset(os.path.join(opt.dataroot, '{}_img'.format(opt.phase)))
        self.lab_dir = os.path.join(opt.dataroot, '{}_lab'.format(opt.phase))

        self.img_transforms = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.5, 0.5, 0.5),
                                                                       (0.5, 0.5, 0.5))])
        self.lab_transform = MaskToTensor()

        self.phase = opt.phase

    def __getitem__(self, index):
        """
        Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            image (tensor) - - an image
            label (tensor) - - its corresponding segmentation
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        img_path = self.img_paths[index]
        lab_path = os.path.join(self.lab_dir, os.path.basename(img_path).split('.')[0] + '.png')

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        lab = cv2.imread(lab_path, cv2.IMREAD_UNCHANGED)

        if len(lab.shape) == 3:
            lab = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)

        w, h = self.opt.load_width, self.opt.load_height
        if w > 0 or h > 0:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
            lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_CUBIC)

        _, lab = cv2.threshold(lab, 127, 255, cv2.THRESH_BINARY)

        if (not self.opt.no_flip) and random.random() > 0.5:
            if random.random() > 0.5:
                img = np.fliplr(img)
                lab = np.fliplr(lab)
            else:
                img = np.flipud(img)
                lab = np.flipud(lab)

        if self.opt.use_augment:
            if random.random() > 0.5:
                angle, scale, shift = get_params()
                img = affine_transform(img, angle, scale, shift, w, h)
                lab = affine_transform(lab, angle, scale, shift, w, h)

        _, lab = cv2.threshold(lab, 127, 1, cv2.THRESH_BINARY)

        img = self.img_transforms(Image.fromarray(img.copy()))
        lab = self.lab_transform(lab.copy()).unsqueeze(0)

        return {'image': img, 'label': lab, 'A_paths': img_path, 'B_paths': lab_path}

    def __len__(self):
        return len(self.img_paths)

