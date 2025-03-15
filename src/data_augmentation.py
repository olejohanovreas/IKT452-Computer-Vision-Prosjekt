import random
import numpy as np
import cv2


def rotate_image(image, rotation_range):
    """
    Randomly rotates an image within +- rotation_range degrees
    :param image:
    :param rotation_range:
    """
    h, w = image.shape[:2]
    angle = random.uniform(-rotation_range, rotation_range)
    M_rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(image, M_rot, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    return rotated


def shift_image(image, shift_range):
    """
    Randomly shifts an image by up to shift_range fraction of width and height
    :param image:
    :param shift_range:
    """
    h, w = image.shape[:2]
    max_dx = shift_range * w
    max_dy = shift_range * h
    dx = random.uniform(-max_dx, max_dx)
    dy = random.uniform(-max_dy, max_dy)
    M_shift = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(image, M_shift, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    return shifted


def flip_image(image, flip_h, flip_v):
    """
    Randomly flips an image horizontally and/or vertically
    :param flip_v:
    :param flip_h:
    :param image:
    """
    flipped = image.copy()
    if random.random() < 0.5 and flip_h:
        flipped = cv2.flip(flipped, 1)
    if random.random() < 0.5 and flip_v:
        flipped = cv2.flip(flipped, 0)
    return flipped


def add_noise(image, noise_stddev):
    """
    Adds gaussian noise withing the specified stddev
    :param image:
    :param noise_stddev:
    """
    noisy = image.copy()
    noise = np.random.normal(0, noise_stddev, noisy.shape).astype(np.float32)
    noisy += noise
    noisy = np.clip(noisy, 0, 1)
    return noisy


def augment_image(image, rotation_range, shift_range, flip_h, flip_v, noise_stddev):
    """
    Augments the image based on the specified parameters
    :param image:
    :param rotation_range:
    :param shift_range:
    :param flip:
    :param noise_stddev:
    """
    augmented = image.copy()
    if rotation_range:
        augmented = rotate_image(augmented, rotation_range)
    if shift_range:
        augmented = shift_image(augmented, shift_range)
    if flip_h or flip_v:
        augmented = flip_image(augmented, flip_h, flip_v)
    if noise_stddev:
        augmented = add_noise(augmented, noise_stddev)
    return augmented
