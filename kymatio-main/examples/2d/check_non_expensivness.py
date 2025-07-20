import numpy as np
import matplotlib.pyplot as plt
from kymatio import Scattering2D
import torch
from torchvision import datasets, transforms
from PIL import Image
import os
import random
import math

# Parameters
n = 100  # Number of random MNIST images to load

# Transform to convert images to tensors
transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

# Load MNIST dataset
mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Select n random indices
random_indices = random.sample(range(len(mnist_data)), n)

# Load images, resize to 32x32, and convert to NumPy arrays
random_images_resized = []
for idx in random_indices:
    image, label = mnist_data[idx]
    random_images_resized.append(image[0])
    # Display the original MNIST image
    # plt.figure(figsize=(2, 2))
    # plt.imshow(image[0], cmap='gray')
    # plt.title('Original MNIST Image')
    # plt.axis('off')
    # plt.show()


# ####################################################################
# # Scattering computations
# #-------------------------------------------------------------------
L = 8
J = 3

scattering = Scattering2D(J=J, shape=random_images_resized[0].shape, L=L, max_order=3, frontend='torch', model_kind = 'invertible_scattering')

images_scattering_coefficients = []
for i in range(n):
    scat_coeffs = scattering(random_images_resized[i])[0]
    images_scattering_coefficients.append(scat_coeffs)


contraction_factors = []
max_diff = 0
avrg_diff = 0
num_of_positive_diffs = 0

for i in range(n):
    for j in range(i):

        image_i = random_images_resized[i]
        image_j = random_images_resized[j]

        coeffs_i = images_scattering_coefficients[i]
        coeffs_j = images_scattering_coefficients[j]

        image_i_minus_j = (image_i - image_j)
        image_i_minus_j_flattend = torch.flatten(image_i_minus_j)

        dist_images = torch.norm( image_i_minus_j , p=2)
        dist_coeffs = torch.norm( torch.flatten( (coeffs_i - coeffs_j) ), p=2 )

        contraction_factors.append(dist_images / dist_coeffs)

        #print(f" dist( imag_{i} , image_{j} ) = {dist_images} , dist( coeffs_{i} , coeffs_{j} ) = {dist_coeffs} )")

        diff = dist_images - dist_coeffs
        
        if diff < 0:
            max_diff = max(max_diff, abs(diff) )
            avrg_diff = avrg_diff + abs(diff)
            num_of_positive_diffs += 1


num_of_samples = (n*(n-1)) /2
avrg_contraction_factor = sum(contraction_factors) / (num_of_samples)
min_contraction_factor = min(contraction_factors)
max_contraction_factor = max(contraction_factors)

if num_of_positive_diffs > 0:
    avrg_diff = avrg_diff / num_of_positive_diffs


print("\n*********************************RESULTS*********************************\n")
print(f" max diff : {max_diff} \n avrg diff : {avrg_diff} \n")
print(f" min contraction factor : {min_contraction_factor} \n max contraction factor : {max_contraction_factor} \n avrg contraction factor : {avrg_contraction_factor} \n")