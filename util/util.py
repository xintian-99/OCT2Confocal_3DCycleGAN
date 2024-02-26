"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
from tifffile import imsave

# def tensor2im(input_image, imtype=np.uint8):
#     """"Converts a Tensor array into a numpy image array.

#     Parameters:
#         input_image (tensor) --  the input image tensor array
#         imtype (type)        --  the desired type of the converted numpy array
#     """
#     if not isinstance(input_image, np.ndarray):
#         if isinstance(input_image, torch.Tensor):  # get the data from a variable
#             image_tensor = input_image.data
#         else:
#             return input_image
#         image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
#         if image_numpy.shape[0] == 1:  # grayscale to RGB
#             image_numpy = np.tile(image_numpy, (3, 1, 1))
#         image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
#     else:  # if it is a numpy array, do nothing
#         image_numpy = input_image
#     return image_numpy.astype(imtype)
def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    #     # Print the shape of the input_image
    # if isinstance(input_image, np.ndarray):
    #     # print(input_image.shape)
    # elif isinstance(input_image, torch.Tensor):
    #     # print(input_image.size())
    # else:
    #     print("Unknown type for input_image")
         
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        
        if len(image_numpy.shape) == 3:
            # This is a 2D image tensor  D, H, W, C 
            if image_numpy.shape[0] == 1:  # grayscale to RGB
                image_numpy = np.tile(image_numpy, (3, 1, 1))
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: transpose and scaling
        
        elif len(image_numpy.shape) == 4:
            # This is a 3D image tensor  D, H, W, C . Assuming that D is the depth (or number of slices).
            # If you want to convert to RGB (3D to 3-channel 2D), this would be more complex and might require averaging or selecting specific slices.
            image_numpy = (np.transpose(image_numpy, (1, 2, 3, 0)) + 1) / 2.0 * 255.0  # post-processing: transpose and scaling
           
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    # print(image_numpy.shape) 
    return image_numpy.astype(imtype)



def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)

def save_4D_image(image_numpy, image_path, aspect_ratio=1.0):
    """
    Save a 4D numpy image to the disk as a multi-page TIFF.

    Parameters:
        image_numpy (numpy array) -- 4D input numpy array (D, H, W, C)
        image_path (str)          -- the path of the image
        aspect_ratio (float)      -- the aspect ratio for resizing (default is 1.0)
    """
    
    D, H, W, C = image_numpy.shape

    # Validate the number of channels
    if C not in [1, 3]:
        raise ValueError("Only grayscale (1 channel) or RGB (3 channels) images are supported.")
    
    # Assuming the aspect ratio adjustment is only applied to H and W
    if aspect_ratio > 1.0:
        new_H = H
        new_W = int(W * aspect_ratio)
    elif aspect_ratio < 1.0:
        new_H = int(H / aspect_ratio)
        new_W = W
    else:
        new_H, new_W = H, W

    # Resize each slice and put in the new array
    resized_image = np.zeros((D, new_H, new_W, C), dtype=image_numpy.dtype)
    for d in range(D):
        for c in range(C):
            slice_pil = Image.fromarray(image_numpy[d, :, :, c])
            slice_resized = slice_pil.resize((new_W, new_H), Image.BICUBIC)
            resized_image[d, :, :, c] = np.array(slice_resized)

    # Save the image as a multi-page TIFF
    if C == 1:  # Grayscale
        resized_image = resized_image.squeeze(-1)  # Remove the channel dimension
    imsave(image_path, resized_image)

def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
