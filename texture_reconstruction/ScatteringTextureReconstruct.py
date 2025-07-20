
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join(current_dir, '..', '..', 'texture_reconstruction', 'scattering')
sys.path.append(relative_path)
import scattering
import random
from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


def load_texture(idx):
    # Path to your local CUReT dataset directory
    dataset_path = os.path.join(current_dir, 'kth_tips_grey_200x200')

    # Collect all image file paths from the dataset directory
    image_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(dataset_path)
        for file in files
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    ]

    # Ensure there are images in the dataset
    if not image_files:
        raise ValueError("No image files found in the specified dataset directory.")

    # Randomly select an image file
    random_image_file = image_files[idx]

    # Load the image using PIL
    image = Image.open(random_image_file)#.convert('RGB')

    # Define a transformation to convert the image to a tensor
    transform = transforms.ToTensor()

    # Apply the transformation to the image
    image_tensor = transform(image)#[0]

    # Now, image_tensor is a PyTorch tensor
    print(f"Randomly selected image tensor shape: {image_tensor.shape}")
    return image_tensor

if __name__ == '__main__':
    shape = (200, 200)  # Example shape of the input image
    J = 2  # Scattering scale
    L = 6  # Number of angles
    m = 2  # Order of scattering coefficients

    output_dir = 'synthesized_textures'
    os.makedirs(output_dir, exist_ok=True)

    # Step 2: Create the Scattering instance
    for i in range(3,10):
      texture_sample = load_texture(i)
      image_syn = scattering.synthesis(estimator_name='s_mean', target=texture_sample, mode='image', L=L)
      # Display the original and synthesized images side by side
      fig, axs = plt.subplots(1, 2, figsize=(10, 5))

      # Original image
      axs[0].imshow(texture_sample[0], cmap='gray')
      axs[0].set_title('Original Texture Image')
      axs[0].axis('off')

      # Synthesized image
      axs[1].imshow(image_syn[0], cmap='gray')
      axs[1].set_title('Synthesized Texture Image')
      axs[1].axis('off')
      plt.tight_layout()
      plot_path = os.path.join(output_dir, f'synthesized_texture_invertible_{i}.png')
      plt.savefig(plot_path, bbox_inches='tight')
      plt.close()

      plt.show()
