import numpy as np
import matplotlib.pyplot as plt
from kymatio import Scattering2D
import torch
from torchvision import datasets, transforms
from PIL import Image
import os
import matplotlib.pyplot as plt





if __name__ == "__main__":

    img_name = os.path.join(os.getcwd(),"kymatio-main","examples","2d","images","digit.png")

    ####################################################################
    # Scattering computations
    #-------------------------------------------------------------------
    # First, we read the input digit:
    image_dim = 64
    image = Image.open(img_name).convert('L').resize((image_dim, image_dim))
    #image = np.array(image)

    # Step 1: Load an MNIST image
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    src_img_tensor = transform(image)[0]


    mnist_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    rnd = np.random.default_rng()
    rnd_idxes = set((rnd.random(10)*100).round().tolist())
    rnd_idxes = [int(idx) for idx in rnd_idxes]
    images = [mnist_dataset[idx][0][0] for idx in rnd_idxes]


    # Step 2: Compute scattering coefficients
    L = 2
    J = 4
    max_order = 3
    scattering = Scattering2D(J=J, shape=images[0].shape, L=L, max_order=max_order, frontend='torch', out_type="list", model_kind='invertible_scattering',downsample=False, dilation_optimization=False)

    ####################################################################
    # We now compute the scattering coefficients:




    #src_img_tensor = image.astype(np.float32) / 255.

    selected_indices = [0,1,2]
    for image_idx , image in enumerate(images):
        plt.figure()
        plt.title(f"Input Image {image_idx}")
        plt.imshow(image.cpu().numpy(), cmap='gray')
        plt.axis('off')
        plt.show()

        coefficients = []
        scat_coeffs = scattering(image, downsample=False) 
                   
        
       
        reconstructed_image = scattering.inverse_scattering(scat_coeffs,last_layer=None)
        reconstructed_image = reconstructed_image[...,0]

        # Plot the reconstructed image
        plt.figure()
        plt.title(f"Reconstructed Image {image_idx}")
        # If reconstructed_image is a torch tensor, convert to numpy
        img_to_plot = reconstructed_image.cpu().detach().numpy()
        plt.imshow(img_to_plot, cmap='gray')
        plt.axis('off')
        plt.show()

