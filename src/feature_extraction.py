import os
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from sklearn.cluster import KMeans


def extract_hog_features(image, hog_config):
    """
    Extract hog features from image
    :param image:
    :param hog_config:
    """
    features = hog(
        image,
        orientations=hog_config['orientations'],
        pixels_per_cell=tuple(hog_config['pixels_per_cell']),
        cells_per_block=tuple(hog_config['cells_per_block']),
        block_norm=hog_config['block_norm'],
        feature_vector=True
    )
    return features


def extract_lbp_features(image, lbp_config):
    """
    Extract lbp features and return the normalized histogram
    :param image:
    :param lbp_config:
    """
    radius = lbp_config['radius']
    n_points = lbp_config['n_points']
    method = lbp_config['method']
    lbp = local_binary_pattern(image, radius, n_points, method)

    # Calculate histogram of lbp values
    n_bins = lbp_config['bins']
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins + 1), density=True)
    return hist


def extract_gabor_features(image, gabor_config):
    """
    Apply gabor filters with various frequencies and orientations and return mean and variance statistics as features
    :param image:
    :param gabor_config:
    """
    features = []
    frequencies = gabor_config['frequencies']
    angles = gabor_config['angles']
    for theta in angles:
        theta_rad = theta * np.pi / 180
        for freq in frequencies:
            kernel = cv2.getGaborKernel(
                ksize=gabor_config['kernel_size'],
                sigma=gabor_config['sigma'],
                theta=theta_rad,
                lambd=gabor_config['lambd'] / freq,
                gamma=gabor_config['gamma'],
                psi=gabor_config['psi']
            )
            filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
            features.append(filtered.mean())
            features.append(filtered.var())
    return np.array(features)


def extract_glcm_features(image, glcm_config):
    """
    Extract texture features using gray-level co-occurence matrix (glcm)
    :param image:
    :param glcm_config:
    """
    levels = glcm_config['levels']
    image_quantized = np.uint8(image / 256 * levels)

    glcm = graycomatrix(
        image_quantized,
        distances=glcm_config['distances'],
        angles=glcm_config['angles'],
        levels=levels,
        symmetric=True,
        normed=True
    )
    properties = glcm_config['properties']
    feats = []
    for prop in properties:
        feat = graycoprops(glcm, prop)
        feats.append(feat.mean())
    return np.array(feats)


class SIFTBoW:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.sift = cv2.SIFT.create()
        self.kmeans = None
        self.vocab = None

    def extract_sift_descriptors(self, image):
        keypoints, descriptors = self.sift.detectAndCompute(image, None)
        if descriptors is None:
            return np.array([])
        return descriptors

    def build_vocabulary(self, images):
        all_descriptors = []
        for img in images:
            desc = self.extract_sift_descriptors(img)
            if desc.size != 0:
                all_descriptors.append(desc)
        all_descriptors = np.vstack(all_descriptors)
        self.kmeans = KMeans(n_clusters=self.vocab_size, random_state=42)
        self.kmeans.fit(all_descriptors)
        self.vocab = self.kmeans.cluster_centers_

    def transform(self, image):
        descriptors = self.extract_sift_descriptors(image)
        if descriptors.size == 0 or self.kmeans is None:
            return np.zeros(self.vocab_size)
        clusters = self.kmeans.predict(descriptors)
        hist, _ = np.histogram(clusters, bins=np.arange(0, self.vocab_size + 1), density=True)
        return hist


def extract_features(path, method, config, progress_bar):
    """
    Extract features from all images in a directory using the specified method
    :param progress_bar:
    :param path:
    :param method:
    :param config:
    """
    features_list = []
    image_paths = []

    if method == 'sift':
        sift_bow = SIFTBoW(config['sift']['vocab_size'])
        images = []
        for file in os.listdir(path):
            filepath = os.path.join(path, file)
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            images.append(image)
            image_paths.append(filepath)
            progress_bar.update(1)

        sift_bow.build_vocabulary(images)
        for image in images:
            feat = sift_bow.transform(image)
            features_list.append(feat)
            progress_bar.update(1)

    else:
        for file in os.listdir(path):
            filepath = os.path.join(path, file)
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            if method == 'hog':
                feat = extract_hog_features(image, config['hog'])
            elif method == 'lbp':
                feat = extract_lbp_features(image, config['lbp'])
            elif method == 'gabor':
                feat = extract_gabor_features(image, config['gabor'])
            elif method == 'glcm':
                feat = extract_glcm_features(image, config['glcm'])
            features_list.append(feat)
            image_paths.append(filepath)
            progress_bar.update(1)

    features_matrix = np.array(features_list)
    return features_matrix, image_paths
