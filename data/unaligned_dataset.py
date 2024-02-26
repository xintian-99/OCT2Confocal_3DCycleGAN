import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import tifffile
import numpy as np
import torch

class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        # print(input_nc)
        # print(output_nc)
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        # depth, height, width, channels
        with tifffile.TiffFile(A_path) as tif:
            A_img = tif.asarray()
        with tifffile.TiffFile(B_path) as tif:
            B_img = tif.asarray()
        #
        # if isinstance(A_img, np.ndarray):
        #     print("my_var is a NumPy array")
        # else:
        #     print("my_var is not a NumPy array")
        # A_img = Image.open(A_path).convert('RGB')
        # B_img = Image.open(B_path).convert('RGB')
        # apply image transformation
        # print(A_img.shape)
        # print(B_img.shape)
        A_=[]
        B_=[]


        #flip or not
        flip_bool = random.choice([True, False])
        r_x = random.randint(0, np.maximum(0, self.opt.load_size - self.opt.crop_size))
        r_y = random.randint(0, np.maximum(0, self.opt.load_size - self.opt.crop_size))
        for sa,sb in zip(A_img,B_img):
            sa=torch.tensor(sa)
            sb=torch.tensor(sb)
            if sa.ndim==2:
                sa=sa.unsqueeze(-1).repeat(1,1,3)
            if sb.ndim==2:
                sb=sb.unsqueeze(-1).repeat(1,1,3)
            sa=sa.permute(2,0,1)
            sb=sb.permute(2,0,1)
            saa= self.transform_A(sa)
            sbb= self.transform_B(sb)

            # # the flip option
            if flip_bool:
                saa = torch.flip(saa, dims=(2,))
                sbb = torch.flip(sbb, dims=(2,))

            if 'crop' in self.opt.preprocess:
                A = A[:, r_x:r_x+self.opt.crop_size, r_y:r_y+self.opt.crop_size]
                B = B[:, r_x:r_x+self.opt.crop_size, r_y:r_y+self.opt.crop_size]
                A_.append(self.transform_A(saa))
                B_.append(self.transform_B(sbb))
        # A = self.transform_A(A_img)
        # B = self.transform_B(B_img)
        # Channel×Number of Images×Height×Width
        A=torch.stack(A_, dim=1)
        B=torch.stack(B_, dim=1)
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
