""" Data operations functions files"""
from __future__ import print_function

import glob
import os
from typing import List, Optional, Tuple

import numpy as np
from skimage import io
import skimage.transform as trans
from keras.preprocessing.image import ImageDataGenerator

Sky = [128, 128, 128]
Building = [128, 0, 0]
Pole = [192, 192, 128]
Road = [128, 64, 128]
Pavement = [60, 40, 222]
Tree = [128, 128, 0]
SignSymbol = [192, 128, 128]
Fence = [64, 64, 128]
Car = [64, 0, 128]
Pedestrian = [64, 64, 0]
Bicyclist = [0, 128, 192]
Unlabelled = [0, 0, 0]

COLOR_DICT = np.array(
    [
        Sky,
        Building,
        Pole,
        Road,
        Pavement,
        Tree,
        SignSymbol,
        Fence,
        Car,
        Pedestrian,
        Bicyclist,
        Unlabelled,
    ]
)


def adjust_data(
    img: np.array, mask: np.array, flag_multi_class: bool, num_class: int
) -> Tuple[np.array, np.array]:
    """
    Adjust mask & images to be normalized depending on the number of classes
    Args:
        img: image as np array
        mask: mask as numpy array
        flag_multi_class: boolean to indicate if multiple class
        num_class: number of classes

    Returns: tuple of adjusted mask & image array

    """
    if flag_multi_class:
        img = img / 255
        mask = mask[:, :, :, 0] if (len(mask.shape) == 4) else mask[:, :, 0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            # for one pixel in the image, find the class in mask and convert it into
            # one-hot vector index = np.where(mask == i) index_mask = (index[0],
            # index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (
            # len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),
            # dtype = np.int64) + i) new_mask[index_mask] = 1
            new_mask[mask == i, i] = 1
        new_mask = (
            np.reshape(
                new_mask,
                (
                    new_mask.shape[0],
                    new_mask.shape[1] * new_mask.shape[2],
                    new_mask.shape[3],
                ),
            )
            if flag_multi_class
            else np.reshape(
                new_mask, (new_mask.shape[0] * new_mask.shape[1], new_mask.shape[2])
            )
        )
        mask = new_mask
    elif np.max(img) > 1:
        img = img / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return img, mask


def train_generator(
    batch_size: int,
    train_path: str,
    image_folder: str,
    mask_folder: str,
    aug_dict: dict,
    image_color_mode: str = "grayscale",
    mask_color_mode: str = "grayscale",
    image_save_prefix: str = "image",
    mask_save_prefix: str = "mask",
    flag_multi_class: bool = False,
    num_class: int = 2,
    save_to_dir: Optional[str] = None,
    target_size: Tuple[int, int] = (256, 256),
    seed: int = 1,
) -> Tuple[np.array, np.array]:
    """
    train generator function, prepare data for training This function can generate
    image and mask at the same time Use the same seed for image_datagen and
    mask_datagen to ensure the transformation for image and mask is the same To
    visualize results of generator, set save_to_dir = "your path" Args: batch_size:
    int for image in batch train_path: train data folder path image_folder: image
    folder path mask_folder: mask folder path aug_dict: dict with augmentation
    variables image_color_mode: color mode for images (grey or color)
    mask_color_mode: color mode for mask (grey or color) image_save_prefix: prefix of
    image name mask_save_prefix: prefix of mask name flag_multi_class: boolean to
    indicate multiclass num_class: number of classes save_to_dir: folder to save data
    target_size: target image size seed: seed number

    Returns: Tuple of image and mask associated

    """
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed,
    )
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed,
    )
    train_gen = zip(image_generator, mask_generator)
    for (img, mask) in train_gen:
        img, mask = adjust_data(img, mask, flag_multi_class, num_class)
        yield img, mask


def test_generator(
    test_path: str,
    num_image: int = 30,
    target_size: Tuple[int, int] = (256, 256),
    flag_multi_class: bool = False,
    as_gray: bool = True,
):
    """
    Test generator function using keras
    Args:
        test_path: path to test data
        num_image: number of images
        target_size: target image size
        flag_multi_class:
        as_gray:

    Returns:

    """
    for i in range(num_image):
        img = io.imread(os.path.join(test_path, f"{i}.png"), as_gray=as_gray)
        img = img / 255
        img = trans.resize(img, target_size)
        img = np.reshape(img, img.shape + (1,)) if (not flag_multi_class) else img
        img = np.reshape(img, (1,) + img.shape)
        yield img


def gene_train_npy(
    image_path: str,
    mask_path: str,
    flag_multi_class: bool = False,
    num_class: int = 2,
    image_prefix: str = "image",
    mask_prefix: str = "mask",
    image_as_gray: bool = True,
    mask_as_gray: bool = True,
) -> Tuple[np.array, np.array]:
    """
    Training generator in numpy
    Args:
        image_path: path to image
        mask_path: path to mask
        flag_multi_class: boolean, true if multiple classes
        num_class: number of classes
        image_prefix: prefix of image name
        mask_prefix: prefix of mask name
        image_as_gray: bool for image in grey if true
        mask_as_gray: bool for mask in grey grey if true

    Returns: two array, image_arr, mask_arr corresponding to

    """
    image_name_arr = glob.glob(os.path.join(image_path, f"{image_prefix}.png"))
    image_arr = []
    mask_arr = []
    for _, item in enumerate(image_name_arr):
        img = io.imread(item, as_gray=image_as_gray)
        img = np.reshape(img, img.shape + (1,)) if image_as_gray else img
        mask = io.imread(
            item.replace(image_path, mask_path).replace(image_prefix, mask_prefix),
            as_gray=mask_as_gray,
        )
        mask = np.reshape(mask, mask.shape + (1,)) if mask_as_gray else mask
        img, mask = adjust_data(img, mask, flag_multi_class, num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr, mask_arr


def label_visualize(num_class: int, color_dict: np.array, img: np.array):
    """
    Visualise label on image
    Args:
        num_class: number of classes
        color_dict: dict correspond to class colors
        img: image to color

    Returns: normalized image with class in.

    """
    img = img[:, :, 0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i, :] = color_dict[i]
    return img_out / 255


def save_result(
    save_path: str,
    npyfile: List[np.array],
    flag_multi_class: bool = False,
    num_class: int = 2,
    output_size: Tuple[int, int] = (256, 256),
):
    """
    Function to save predicted result as images
    Args:
        save_path: path to folder to save data in
        npyfile: Numpy array(s) of predictions
        flag_multi_class: activate if multiclass
        num_class: number of classes
        output_size: image output size (used to resize)

    Returns: None

    """
    for i, item in enumerate(npyfile):
        img = (
            label_visualize(num_class, COLOR_DICT, item)
            if flag_multi_class
            else item[:, :, 0]
        )
        # resize image to fit input image size
        img = trans.resize(img, output_size)
        io.imsave(os.path.join(save_path, f"{i}_predict.png"), img)
