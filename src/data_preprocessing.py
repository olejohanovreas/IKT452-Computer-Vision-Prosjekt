import cv2
import numpy as np


def load_image(image_path):
    """
    Loads an image and converts it to grayscale.
    :param image_path:
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image


def crop_image(image, threshold):
    """
    Crops the image using a threshold.
    :param image:
    :param threshold:
    """
    # Apply thresholding and find contours
    _, thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Return the original image if no contours are found
    if not contours:
        return image

    # Combine all contours into one and compute the bounding box. Then crop the image with this BB.
    all_contours = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_contours)
    cropped = image[y:y + h, x:x + w]
    return cropped


def resize_image(image, size):
    """
    Resizes the image to the given size.
    :param image:
    :param size:
    """
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return resized


def normalize_image(image):
    """
    Normalizes the image to [0,1] range.
    :param image:
    """
    normalized = image.astype(np.float32) / 255.0
    return normalized


def preprocess_image(image_path, size, threshold):
    """
    Preprocesses the given image.
    :param image_path:
    :param size:
    :param threshold:
    """
    image = load_image(image_path)
    image = crop_image(image, threshold)
    image = resize_image(image, size)
    image = normalize_image(image)
    return image
