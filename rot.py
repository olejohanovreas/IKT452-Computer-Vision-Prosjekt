import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Dataset directories
training_dir = "data/Training"
testing_dir = "data/Testing"

# Processed images directories
processed_training_dir = "data/Processed/Training"
processed_testing_dir = "data/Processed/Testing"

# Class names
classes = ['glioma', 'meningioma', 'notumor']

# Ensure processed output directories exist
def create_processed_dirs(base_output_dir):
    for class_name in classes:
        class_path = os.path.join(base_output_dir, class_name)
        os.makedirs(class_path, exist_ok=True)

create_processed_dirs(processed_training_dir)
create_processed_dirs(processed_testing_dir)

def load_images_from_class(dataset_path, class_name, num_samples=3):
    """
    Loads up to num_samples images from the specified class folder in dataset_path.
    Returns a list of tuples: (image_file_path, image_array)
    """
    class_dir = os.path.join(dataset_path, class_name)
    image_files = [os.path.join(class_dir, f) for f in os.listdir(class_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files = image_files[:num_samples]
    images = []
    for image_file in image_files:
        img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append((image_file, img))
    return images

def plot_samples(dataset_path, preprocess, target_size, threshold_value):
    """
    Plots sample images from each class.
    """
    plt.style.use('default')
    fig, axs = plt.subplots(len(classes), 3, figsize=(10, 10))
    for i, class_name in enumerate(classes):
        images = load_images_from_class(dataset_path, class_name, num_samples=3)
        for j, (img_path, img) in enumerate(images):
            if preprocess:
                img = crop_and_resize(img, target_size=target_size, threshold_value=threshold_value)
            axs[i, j].imshow(img, cmap='gray')
            axs[i, j].axis('off')
            axs[i, j].set_title(f"{class_name}\n{os.path.basename(img_path)}\nSize: {img.shape}")
    plt.tight_layout()
    plt.show()

def get_image_size_statistics(dataset_path):
    """
    Computes and prints average dimensions for each class in the specified dataset_path.
    """
    for class_name in classes:
        class_dir = os.path.join(dataset_path, class_name)
        image_files = [os.path.join(class_dir, f) for f in os.listdir(class_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        sizes = []
        for image_file in image_files:
            img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                sizes.append(img.shape)
        if sizes:
            sizes = np.array(sizes)
            avg_height = np.mean(sizes[:, 0])
            avg_width = np.mean(sizes[:, 1])
            print(f"Class '{class_name}': {len(sizes)} images, Average Size: {avg_height:.2f} x {avg_width:.2f}")
        else:
            print(f"Class '{class_name}': No images found.")

def crop_and_resize(img, target_size, threshold_value):
    """
    Dynamically crops the image by removing excess background based on a threshold and resizes the image.
    """
    # Threshold the image to create a binary mask.
    _, thresh = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)

    # Find the coordinates of all non-zero pixels.
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        # Get the bounding box of the non-zero pixels.
        x, y, w, h = cv2.boundingRect(coords)
        # Crop the image to this bounding box.
        cropped = img[y: y + h, x: x + w]
        return cv2.resize(cropped, target_size)
    else:
        # If no foreground was found, just return the resized original image.
        return cv2.resize(img, target_size)

def process_and_save_images(input_dir, output_dir, target_size, threshold_value):
    """
    Processes all images from input_dir by cropping and resizing them, then saves the processed images in output_dir.
    The directory structure in output_dir will mirror that of input_dir.
    """
    for class_name in classes:
        input_class_dir = os.path.join(input_dir, class_name)
        output_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)
        image_files = [f for f in os.listdir(input_class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for file in image_files:
            input_path = os.path.join(input_class_dir, file)
            img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            processed_img = crop_and_resize(img, target_size, threshold_value)
            output_path = os.path.join(output_class_dir, file)
            cv2.imwrite(output_path, processed_img)

# Display original training data statistics and samples
print("\nOriginal Training Data Statistics:")
get_image_size_statistics(training_dir)
print("\nOriginal Training Data Samples:")
plot_samples(training_dir, preprocess=False, target_size=None, threshold_value=None)

# Process and save images for training and testing datasets
process_and_save_images(training_dir, processed_training_dir, target_size=(224, 224), threshold_value=50)
process_and_save_images(testing_dir, processed_testing_dir, target_size=(224, 224), threshold_value=50)

# Display processed training and testing samples for verification
print("\nProcessed Training Data Samples:")
plot_samples(processed_training_dir, preprocess=False, target_size=None, threshold_value=None)
print("\nProcessed Testing Data Samples:")
plot_samples(processed_testing_dir, preprocess=False, target_size=None, threshold_value=None)
