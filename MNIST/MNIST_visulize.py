import numpy as np
import struct
import matplotlib.pyplot as plt
import os

# Path configuration (update these paths to match your file locations)
FILE_PATHS = {
    'train_images': 'MNIST_DATASET/train-images-idx3-ubyte/train-images-idx3-ubyte',
    'train_labels': 'MNIST_DATASET/train-labels-idx1-ubyte/train-labels-idx1-ubyte',
    'test_images': 'MNIST_DATASET/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte',
    'test_labels': 'MNIST_DATASET/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
}

def read_images(file_path):
    """Read MNIST image data from binary file"""
    with open(file_path, 'rb') as f:
        magic, num_images = struct.unpack('>II', f.read(8))
        rows, cols = struct.unpack('>II', f.read(8))
        
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic} for image file")
            
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        images = image_data.reshape(num_images, rows, cols)
    
    return images

def read_labels(file_path):
    """Read MNIST label data from binary file"""
    with open(file_path, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic} for label file")
            
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    
    return labels

def load_mnist():
    """Load entire MNIST dataset"""
    # Check if files exist
    for path in FILE_PATHS.values():
        if not os.path.exists(path):
            raise FileNotFoundError(f"MNIST file not found: {path}")
    
    # Load data
    train_images = read_images(FILE_PATHS['train_images'])
    train_labels = read_labels(FILE_PATHS['train_labels'])
    test_images = read_images(FILE_PATHS['test_images'])
    test_labels = read_labels(FILE_PATHS['test_labels'])
    
    # Normalize pixel values to [0, 1]
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0
    
    return (train_images, train_labels), (test_images, test_labels)

def visualize_sample(images, labels, num_samples=5):
    """Visualize random samples from the dataset"""
    indices = np.random.choice(len(images), num_samples)
    
    plt.figure(figsize=(10, 2))
    for i, idx in enumerate(indices):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(images[idx], cmap='gray')
        plt.title(f"Label: {labels[idx]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Load the dataset
    (train_images, train_labels), (test_images, test_labels) = load_mnist()
    
    # Print dataset information
    print("Training data:")
    print(f"Images: {train_images.shape} (dtype: {train_images.dtype})")
    print(f"Labels: {train_labels.shape}\n")
    
    print("Test data:")
    print(f"Images: {test_images.shape} (dtype: {test_images.dtype})")
    print(f"Labels: {test_labels.shape}\n")
    
    # Visualize random samples
    print("Training samples:")
    visualize_sample(train_images, train_labels)
    
    print("Test samples:")
    visualize_sample(test_images, test_labels)