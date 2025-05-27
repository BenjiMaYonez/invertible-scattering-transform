import numpy as np
import matplotlib.pyplot as plt
from kymatio import Scattering2D
import torch
from torchvision import datasets, transforms
from PIL import Image
import os






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
    J = 3
    max_order = 3
    scattering = Scattering2D(J=J, shape=images[0].shape, L=L, max_order=max_order, frontend='torch', out_type="list", model_kind='invertible_scattering',downsample=False)

    ####################################################################
    # We now compute the scattering coefficients:




    #src_img_tensor = image.astype(np.float32) / 255.

    selected_indices = [0,1,2]
    for image_idx , image in enumerate(images):
        coefficients = []
        scat_coeffs = scattering(image, downsample=False) 
                   
        reconstructed_image = scattering.inverse_scattering(scat_coeffs,last_layer=None)

        ####################################################################
        # There are 127 scattering coefficients, among which 1 is low-pass, $JL=18$ are of first-order and $L^2(J(J-1)/2)=108$
        # are of second-order. Due to the subsampling by $2^J=8$, the final spatial grid is of size $4\times4$.
        # We now retrieve first-order and second-order coefficients for the display.
        # len_order_1 = J*L
        # scat_coeffs_order_1 = scat_coeffs[1:1+len_order_1, :, :]

        # len_order_2 = (J*(J-1)//2)*(L**2)
        # scat_coeffs_order_2 = scat_coeffs[1+len_order_1:, :, :]

        # coefficients = scat_coeffs_order_2
        #coefficients = scat_coeffs

        # window_rows, window_columns = coefficients[0].shape

        # num_coefficients = len(coefficients)
        # print("Number of coefficients (channels):", num_coefficients)

        # # For demonstration, let's select a few coefficients
        # selected_indices = [0, num_coefficients // 2, num_coefficients - 1]  # First, middle, last coefficients

        # # Step 4: Analyze each selected coefficient
        # for idx in selected_indices:
        #     coeff = coefficients[idx]
        #     visualize_frequencies(coeff, max_order, idx)

