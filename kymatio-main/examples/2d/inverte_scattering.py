import numpy as np
import matplotlib.pyplot as plt
from kymatio import Scattering2D
import torch
from torchvision import datasets, transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from  plot_scattering_frequencies import visualize_frequencies




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
        transforms.Resize((image_dim, image_dim)),
        transforms.ToTensor()
    ])
    src_img_tensor = transform(image)[0]


    mnist_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    rnd = np.random.default_rng()
    rnd_idxes = set((rnd.random(10)*100).round().tolist())
    rnd_idxes = [int(idx) for idx in rnd_idxes]
    # images = [mnist_dataset[idx][0][0] for idx in rnd_idxes]

####### unit testing ###########################################
    M , N = 10, 10
    images = [torch.zeros(M,N) for _ in range(3)]
    images[0] = torch.zeros(M,N)
    #initing images[1] so all values are 1.0
    images[1][0:M, 0:N] = 1.0
    #initing images[2] to have ones in the middle  and zeros elsewhere
    images[2] = torch.zeros(M,N)
    images[2][M//4:M//4*3, N//4:N//4*3] = 1.0
###########################################

    # Step 2: Compute scattering coefficients
    
    L = 2
    J = 3
    max_order = 2
    scattering = Scattering2D(J=J, shape=images[0].shape, L=L, max_order=max_order, frontend='torch', out_type="list", model_kind='invertible_scattering',downsample=False, dilation_optimization=False)

    ####################################################################
    # We now compute the scattering coefficients:




    #src_img_tensor = image.astype(np.float32) / 255.



    selected_indices = [0,1,2]
    for image_idx , image in enumerate(images):
        plt.figure()
        plt.title(f"Input Image {image_idx}")
        plt.imshow(image.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        plt.colorbar()
        plt.axis('off')
        plt.show()

        coefficients = []
        scat_coeffs = scattering(image, downsample=False) 
        print("finished computing scattering coefficients")
        for i in range(max_order+1):
            coeff = [coeff for coeff in scat_coeffs if coeff['depth'] == i][0]  # Get the first coefficient of depth i
            visualize_frequencies(coeff['coef'].squeeze().numpy(), coeff['depth'], coeff['n'], image_idx)
        
       

        reconstructed_image = scattering.inverse_scattering(scat_coeffs,last_layer=None)
        reconstructed_image = reconstructed_image[...,0]

        # Plot the reconstructed image
        plt.figure()
        plt.title(f"Reconstructed Image {image_idx}")
        # If reconstructed_image is a torch tensor, convert to numpy
        img_to_plot = reconstructed_image.cpu().detach().numpy()
        plt.imshow(img_to_plot, cmap='gray', vmin=0, vmax=1)
        plt.colorbar()
        plt.axis('off')
        plt.show()

